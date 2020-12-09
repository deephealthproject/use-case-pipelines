from models.unet import UNet
import argparse
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torchvision import transforms
from yaml_segmentation_dataset import MSSegmentationDataset
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from loss import dice
from torch import optim
from tqdm import tqdm


def main(args):
    # training_transforms = transforms.Compose([
    #     transforms.RandomResizedCrop(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(60.243489265441895, 167.91686515808107),
    # ])
    train_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(60.243489265441895, 167.91686515808107),
            ToTensorV2(),
        ]
    )
    train_dataset = MSSegmentationDataset(dataset=args.in_ds, transform=train_transform, split=['training'])
    valid_dataset = MSSegmentationDataset(dataset=args.in_ds, transform=train_transform, split=['validation'])
    test_dataset = MSSegmentationDataset(dataset=args.in_ds, split=['test'])
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, drop_last=True,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False,
                                 pin_memory=True)

    dataloaders = {"train": train_dataloader, "valid": valid_dataloader, 'test': test_dataloader}
    device = torch.device('cpu' if not args.gpu else 'cuda')

    # Model, loss, optimizer
    model = UNet(in_channels=1, out_channels=1)
    if torch.cuda.device_count() > 1 and args.gpu:
        model = torch.nn.DataParallel(model, device_ids=np.where(np.array(args.gpu) == 1)[0])
    model.to(device)

    dsc_loss = dice.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_train = []
    loss_valid = []
    best_validation_dsc = 0.0
    step = 0
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            validation_pred = []
            validation_true = []

            for i, (x, gt) in enumerate(dataloaders[phase]):
                if phase == "train":
                    step += 1

                x, gt = x.to(device), gt.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    pred = model(x)

                    loss = dsc_loss(pred, gt)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        pred_np = pred.detach().cpu().numpy()
                        validation_pred.extend([pred_np[s] for s in range(pred_np.shape[0])])
                        gt_np = gt.detach().cpu().numpy()
                        validation_true.extend([gt_np[s] for s in range(gt_np.shape[0])])
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                                # logger.image_list_summary(
                                #     tag,
                                #     log_images(x, y_true, y_pred)[:num_images],
                                #     step,
                                # )
                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    print(i, ' loss ', loss.item())
            if phase == "valid":
                # Mean Dice TODO
                # if mean_dsc > best_validation_dsc:
                #     best_validation_dsc = mean_dsc
                #     torch.save(model.state_dict(), os.path.join(args.weights, "unet.pth"))
                loss_valid = []


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
    parser.add_argument('--ckpts', type=str)
    main(parser.parse_args())
