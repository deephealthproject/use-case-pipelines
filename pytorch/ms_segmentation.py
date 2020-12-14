from models.unet import UNet, Nabla
import argparse
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from yaml_segmentation_dataset import MSSegmentationDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from loss import dice
from torch import optim
from tqdm import tqdm
from PIL import Image

def Threshold(a, thresh=0.5):
    a[a >= thresh] = 1
    a[a < thresh] = 0
    return a

class Evaluator:
    def __init__(self):
        self.eps = 1e-06
        self.buf = []

    def ResetEval(self):
        self.buf = []

    def DiceCoefficient(self, img, gt, thresh=0.5):
        img = Threshold(img, thresh)
        gt = Threshold(gt, thresh)
        intersection = np.logical_and(img, gt).sum()
        rval = (2 * intersection + self.eps) / (img.sum() + gt.sum() + self.eps)

        self.buf.append(rval)
        return rval

    def MeanMetric(self):
        if not self.buf:
            return 0
        return sum(self.buf) / len(self.buf)

def main(args):
    writer = SummaryWriter(comment='_nabla_bce')
    train_transform = A.Compose(
        [
            A.Resize(args.size, args.size),
            A.Normalize(60.243489265441895, 167.91686515808107),
            ToTensorV2(),
        ]
    )
    valid_transform = A.Compose(
        [
            A.Resize(args.size, args.size),
            A.Normalize(60.243489265441895, 167.91686515808107),
            ToTensorV2(),
        ]
    )
    train_dataset = MSSegmentationDataset(dataset=args.in_ds, transform=train_transform, split=['training'])
    valid_dataset = MSSegmentationDataset(dataset=args.in_ds, transform=valid_transform, split=['validation'])
    test_dataset = MSSegmentationDataset(dataset=args.in_ds, transform=valid_transform, split=['test'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                                 pin_memory=True)

    dataloaders = {"train": train_dataloader, "valid": valid_dataloader, 'test': test_dataloader}
    device = torch.device('cpu' if not args.gpu else 'cuda')

    # Model, loss, optimizer
    model = Nabla(in_channels=1, out_channels=args.num_classes)
    if torch.cuda.device_count() > 1 and args.gpu:
        model = torch.nn.DataParallel(model, device_ids=np.where(np.array(args.gpu) == 1)[0])
    model.to(device)

    dsc_loss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.ckpts is None:
        best_validation_dsc = 0.0
        load_epoch = 0
    else:
        checkpoint = torch.load(args.ckpts)
        model.load_state_dict(checkpoint['state_dict'])
        load_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_validation_dsc = checkpoint['best_metric']
        print("Loaded checkpoint epoch ", load_epoch, " with best metric ", best_validation_dsc)

    train_eval = Evaluator()
    valid_eval = Evaluator()
    for epoch in tqdm(range(load_epoch, args.epochs), total=args.epochs):
        print()
        loss_train = []
        loss_valid = []
        train_eval.ResetEval()
        valid_eval.ResetEval()
        for phase in ["train", "valid"]:
            print('STARTING ', phase)
            if phase == "train":
                model.train()
            else:
                model.eval()

            for i, (x, gt, names) in enumerate(dataloaders[phase]):
                pred_list = []
                gt_list = []
                if phase == "train":
                    optimizer.zero_grad()

                if isinstance(dsc_loss, torch.nn.CrossEntropyLoss):
                    x, gt = x.to(device), gt.to(device, dtype=torch.long)
                    gt = gt.squeeze()
                else:
                    x, gt = x.to(device), gt.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    pred = model(x)
                    loss = dsc_loss(pred, gt)
                    pred = torch.nn.Sigmoid()(pred)

                    pred_np = pred.detach().cpu().numpy()
                    pred_list.extend([pred_np[s] for s in range(pred_np.shape[0])])
                    gt_np = gt.detach().cpu().numpy()
                    gt_list.extend([gt_np[s] for s in range(gt_np.shape[0])])

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        loss_train.append(loss.item())
                        for j, (p, g) in enumerate(zip(pred_list, gt_list)):
                            train_eval.DiceCoefficient(p, g)

                    elif phase == "valid":
                        loss_valid.append(loss.item())
                        for j, (p, g) in enumerate(zip(pred_list, gt_list)):
                            valid_eval.DiceCoefficient(p, g)
                            if args.out_dir is not None:
                                p *= 255
                                g *= 255
                                pil = Image.fromarray(p[0].astype('uint8'))
                                pil.save(os.path.join(args.out_dir, f'{names[j]}.png'))
                                pil = Image.fromarray(g[0].astype('uint8'))
                                pil.save(os.path.join(args.out_dir, f'{names[j]}_gt.png'))

        mean_train_dsc = train_eval.MeanMetric()
        print('Mean metric train epoch ', epoch, ': ', mean_train_dsc)

        mean_valid_dsc = valid_eval.MeanMetric()
        print('Mean metric validation epoch ', epoch, ': ', mean_valid_dsc)
        if mean_valid_dsc > best_validation_dsc:
            best_validation_dsc = mean_valid_dsc
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_validation_dsc
            }
            torch.save(state, os.path.join(args.weights, "nabla_epoch_" + str(epoch) + ".pth"))

        loss_train = np.mean(loss_train)
        loss_valid = np.mean(loss_valid)
        writer.add_scalar('Loss/train', loss_train, epoch)
        writer.add_scalar('Metric/train', mean_train_dsc, epoch)
        writer.add_scalar('Loss/validation', loss_valid, epoch)
        writer.add_scalar('Metric/validation', mean_valid_dsc, epoch)
        print('Mean loss_train epoch ', epoch, ': ', loss_train)
        print('Mean loss_valid epoch ', epoch, ': ', loss_valid)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--epochs', type=int, metavar='INT', default=100)
    parser.add_argument('--batch_size', type=int, metavar='INT', default=16)
    parser.add_argument('--num_classes', type=int, metavar='INT', default=1)
    parser.add_argument('--n_channels', type=int, metavar='INT', default=1,
                        help='Number of slices to stack together and use as input')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--size', type=int, metavar='INT', default=256, help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    parser.add_argument('--weights', type=str, default='weights', help='save weights in this directory')
    parser.add_argument('--ckpts', type=str)
    main(parser.parse_args())
