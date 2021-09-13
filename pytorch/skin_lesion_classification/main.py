import argparse
import os
import random
import sys

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

from dataset import YAMLClassificationDataset
from plots import plot_confusion_matrix

seed = 50
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_dist = np.asarray([3322, 10175, 2423, 627, 2024, 149, 163, 448], dtype=np.float32)
valid_dist = np.asarray([200, 450, 150, 40, 100, 15, 15, 30], dtype=np.float32)
test_dist = np.asarray([1000, 2250, 750, 200, 500, 75, 75, 150], dtype=np.float32)

normalization_isic = (0.6681, 0.5301, 0.5247), (0.1337, 0.1480, 0.1595)
normalization_imagenet = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_weights():
    # calculate class weights to artificially change the class probability distribution
    train_sampling_prob = train_dist / train_dist.sum()
    target_prob = np.ones_like(train_dist, dtype=np.float32) / len(train_dist)
    return target_prob / train_sampling_prob


def get_weights_median():
    dataset_freq = np.array([4522, 12875, 3323, 867, 2624, 239, 253, 628], dtype=np.float32)
    median = np.median(dataset_freq)
    return median / dataset_freq


def SkinLesionModel(model, pretrained=True):
    models_zoo = {
        'resnet18': models.resnet18(pretrained=pretrained),
        'resnet50': models.resnet50(pretrained=pretrained),
        'resnet101': models.resnet101(pretrained=pretrained),
        'resnet152': models.resnet152(pretrained=pretrained),
        'resnext50_32x4d': torch.hub.load('pytorch/vision:v0.8.2', 'resnext50_32x4d', pretrained=pretrained,
                                          verbose=False),
    }
    net = models_zoo.get(model)
    if net is None:
        raise Warning("Wrong Net Name!!")
    return net


def main(args):
    writer = SummaryWriter(comment=args.exp_name)
    os.makedirs(args.weights, exist_ok=True)

    train_transform = iaa.Sequential([
        iaa.Resize((args.size, args.size)),
        iaa.Fliplr(p=0.5),
        iaa.Flipud(p=0.5),
        iaa.Rotate(rotate=(-180, 180)),
        iaa.AdditivePoissonNoise(lam=(0, 10.,)),
        iaa.GammaContrast(gamma=(.5, 1.5)),
        iaa.GaussianBlur(sigma=(.0, .8)),
        iaa.Sometimes(0.25, iaa.CoarseDropout(p=(0, 0.03), size_percent=(0, 0.05))),
    ])

    valid_transform = iaa.Sequential([
        iaa.Resize((args.size, args.size)),
    ])

    train_dataset = YAMLClassificationDataset(dataset=args.in_ds, transform=train_transform, split=['training'],
                                              normalization=normalization_isic)
    valid_dataset = YAMLClassificationDataset(dataset=args.in_ds, transform=valid_transform, split=['validation'],
                                              normalization=normalization_isic)
    test_dataset = YAMLClassificationDataset(dataset=args.in_ds, transform=valid_transform, split=['test'],
                                             normalization=normalization_isic)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 drop_last=False)

    dataloaders = {"train": train_dataloader, "valid": valid_dataloader, 'test': test_dataloader}
    device = torch.device('cpu' if not args.gpu else 'cuda')

    # Model, loss, optimizer
    print('Loading model...')
    model = SkinLesionModel(args.model)

    if args.onnx_export:
        # export onnx
        dummy_input = torch.ones(4, 3, args.size, args.size, device='cpu')
        model.train()
        torch.onnx.export(model, dummy_input, f'{args.model}.onnx', verbose=True, export_params=True,
                          training=torch.onnx.TrainingMode.TRAINING,
                          opset_version=12,
                          do_constant_folding=False,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})

    # Change last linear layer
    model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)

    if torch.cuda.device_count() > 1 and args.gpu:
        model = torch.nn.DataParallel(model, device_ids=np.where(np.array(args.gpu) == 1)[0])
    print(f'Move model to {device}')
    model = model.to(device)

    # loss_fn = nn.modules.loss.CrossEntropyLoss(weight=torch.from_numpy(get_weights()).to(device))
    loss_fn = nn.modules.loss.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    if args.ckpts is None:
        best_valid_acc = 0.
        load_epoch = 0
    else:
        checkpoint = torch.load(args.ckpts)
        model.load_state_dict(checkpoint['state_dict'])
        load_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_valid_acc = checkpoint['best_metric']
        print("Loaded checkpoint epoch ", load_epoch, " with best metric ", best_valid_acc)

    train_acc = 0
    valid_acc = 0
    print('Starting training')
    for epoch in range(load_epoch, args.epochs):
        loss_train = []
        loss_valid = []
        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            correct = 0
            total = 0
            pred_list = []
            gt_list = []
            with tqdm(desc=f"{phase} {epoch}/{args.epochs}", unit="batch", total=len(dataloaders[phase]),
                      file=sys.stdout) as pbar:
                for i, (x, gt, names) in enumerate(dataloaders[phase]):
                    # torchvision.utils.save_image(x, f'batch_{i}.jpg')
                    x, gt = x.to(device), gt.to(device)
                    with torch.set_grad_enabled(phase == "train"):
                        pred = model(x)
                        loss = loss_fn(pred, gt)
                        loss_item = loss.item()
                        pred = torch.nn.functional.softmax(pred, dim=1)

                        pred_np = pred.detach().cpu().numpy()
                        pred_np = pred_np.argmax(axis=1)
                        pred_list.extend(pred_np)
                        gt_np = gt.detach().cpu().numpy()
                        gt_list.extend(gt_np)

                        correct += (pred_np == gt_np).sum()
                        total += pred_np.shape[0]

                        if phase == "train":
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_train.append(loss_item)

                        elif phase == "valid":
                            loss_valid.append(loss_item)

                        pbar.set_postfix(loss=loss_item, accuracy=correct / total)
                        pbar.update()

            accuracy = correct / total
            cm = confusion_matrix(np.array(pred_list).reshape(-1), np.array(gt_list).reshape(-1))
            print(f'{phase} {epoch}/{args.epochs}: accuracy={accuracy:.4f}')
            fig = plt.figure(figsize=(args.num_classes, args.num_classes))
            plot_confusion_matrix(cm, [0, 1, 2, 3, 4, 5, 6, 7])
            writer.add_figure(f'{phase}/confusion', fig, epoch)

            if phase == 'train':
                train_acc = accuracy
                writer.add_scalar(f'{phase}/loss', np.mean(loss_train), epoch)
                writer.add_scalar(f'{phase}/accuracy', train_acc, epoch)

            else:
                valid_acc = accuracy
                writer.add_scalar(f'{phase}/loss', np.mean(loss_valid), epoch)
                writer.add_scalar(f'{phase}/accuracy', valid_acc, epoch)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_metric': best_valid_acc
            }
            torch.save(state, os.path.join(args.weights, f'{args.model}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--model', type=str, help='model name', default='resnet18')
    parser.add_argument('--epochs', type=int, metavar='INT', default=100)
    parser.add_argument('--batch_size', type=int, metavar='INT', default=24)
    parser.add_argument('--num_classes', type=int, metavar='INT', default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--size', type=int, metavar='INT', default=224, help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    parser.add_argument('--weights', type=str, default='weights', help='save weights in this directory')
    parser.add_argument('--ckpts', type=str)
    parser.add_argument('--onnx-export', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='')
    main(parser.parse_args())
