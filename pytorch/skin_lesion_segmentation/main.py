import argparse
import os
import random
import sys

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import YAMLSegmentationDataset

seed = 50
os.environ['PL_GLOBAL_SEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ISIC mean and std calculated on training and validation sets
normalization_isic_seg = (0.67501814, 0.5663187, 0.52339128), (0.11092593, 0.10669603, 0.119005)
normalization_imagenet = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def SkinLesionModel(model, pretrained=True):
    models_zoo = {
        'deeplabv3plus': smp.DeepLabV3Plus('resnet101', encoder_weights='imagenet', aux_params=None),
        'deeplabv3plus_resnext': smp.DeepLabV3Plus('resnext101_32x8d', encoder_weights='imagenet', aux_params=None),
        'pspnet': smp.PSPNet('resnet101', encoder_weights='imagenet', aux_params=None),
        'unetplusplus': smp.UnetPlusPlus('resnet101', encoder_weights='imagenet', aux_params=None),
    }
    net = models_zoo.get(model)
    if net is None:
        raise Warning('Wrong Net Name!!')
    return net


def Denormalize(x, mean, std):
    for i in range(x.shape[1]):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def Upsampling(pred, gt, original_shape):
    # max_shape = torch.max(original_shape, dim=0)
    # pred = F.interpolate(pred, (max_shape, max_shape), mode='bilinear')
    # gt = F.interpolate(gt, (max_shape, max_shape), mode='bilinear')
    # # Cut the image removing black padding
    # ax, ay = (max_shape - pred.shape[2]) // 2, (max_shape - pred.shape[3]) // 2
    # new_pred = pred[:, :, ay:pred.shape[2] + ay, ax:ax + pred.shape[3]]
    # new_gt = gt[:, :, ay:pred.shape[2] + ay, ax:ax + pred.shape[3]]
    # return new_pred, new_gt
    pass


def main(args):
    writer = SummaryWriter(comment=args.exp_name)
    os.makedirs(args.weights, exist_ok=True)

    norm_mean, norm_std = normalization_isic_seg
    train_transform = A.Compose([
        A.Resize(args.size, args.size, cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(-180, 180), interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, value=0,
                 mask_value=0, p=1),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0, rotate_limit=0, interpolation=cv2.INTER_CUBIC,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(0.95, 1.25), rotate_limit=0, interpolation=cv2.INTER_CUBIC,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        A.GaussianBlur(blur_limit=(3, 5)),
        A.GaussNoise(var_limit=(10, 40)),
        A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.CoarseDropout(),
        A.Normalize(norm_mean, norm_std),
        ToTensorV2(),
    ])
    valid_test_transform = A.Compose([
        A.Resize(args.size, args.size, cv2.INTER_CUBIC),
        A.Normalize(norm_mean, norm_std),
        ToTensorV2(),
    ])

    train_dataset = YAMLSegmentationDataset(dataset=args.in_ds, transform=train_transform, split=['training'])
    valid_dataset = YAMLSegmentationDataset(dataset=args.in_ds, transform=valid_test_transform, split=['validation'])
    test_dataset = YAMLSegmentationDataset(dataset=args.in_ds, transform=valid_test_transform, split=['test'], )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  drop_last=False, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 drop_last=False, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'valid': valid_dataloader, 'test': test_dataloader}
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

    if torch.cuda.device_count() > 1 and args.gpu:
        model = torch.nn.DataParallel(model, device_ids=np.where(np.array(args.gpu) == 1)[0])
    print(f'Move model to {device}')
    model = model.to(device)

    # loss_fn = nn.modules.loss.BCEWithLogitsLoss()
    loss_fn = smp.losses.DiceLoss('binary')
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    metric_fn = smp.utils.metrics.IoU(threshold=0.5)

    if args.ckpts is None:
        best_valid_iou = 0.
        load_epoch = 0
    else:
        checkpoint = torch.load(args.ckpts)
        model.load_state_dict(checkpoint['state_dict'])
        load_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_valid_iou = checkpoint['best_metric']
        print('Loaded checkpoint epoch ', load_epoch, ' with best metric ', best_valid_iou)

    if args.train:
        valid_iou = 0
        print('Starting training')
        for epoch in range(load_epoch, args.epochs):
            loss_train = []
            loss_valid = []
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                iou_scores = np.array([])
                with tqdm(desc=f'{phase} {epoch}/{args.epochs - 1}', unit='batch', total=len(dataloaders[phase]),
                          file=sys.stdout) as pbar:
                    for i, (x, gt, original_shape) in enumerate(dataloaders[phase]):
                        # torchvision.utils.save_image(x, f'batch_{i}.jpg')
                        x, gt = x.to(device), gt.to(device, dtype=torch.float32)
                        with torch.set_grad_enabled(phase == 'train'):
                            pred = model(x)
                            loss = loss_fn(pred, gt)
                            loss_item = loss.item()
                            pred, gt = torch.sigmoid(pred.detach()), gt.detach()

                            if phase == 'train':
                                optimizer.zero_grad()
                                loss.backward()
                                optimizer.step()
                                loss_train.append(loss_item)
                            elif phase == 'valid':
                                loss_valid.append(loss_item)
                                # Upsampling(pred, gt, original_shape)

                            iou = metric_fn(pred, gt).item()
                            iou_scores = np.append(iou_scores, iou)

                            pbar.set_postfix(loss=loss_item, IoU=iou_scores.mean())
                            pbar.update()
                            pbar.refresh()

                iou = iou_scores.mean()

                if phase == 'train':
                    train_iou = iou
                    writer.add_scalar(f'{phase}/loss', np.mean(loss_train), epoch)
                    writer.add_scalar(f'{phase}/iou', train_iou, epoch)
                    writer.add_images(f'{phase}/images', Denormalize(x, norm_mean, norm_std), epoch)
                    writer.add_images(f'{phase}/prediction', pred, epoch)
                else:
                    valid_iou = iou
                    writer.add_scalar(f'{phase}/loss', np.mean(loss_valid), epoch)
                    writer.add_scalar(f'{phase}/iou', valid_iou, epoch)
                    grid = torchvision.utils.make_grid(Denormalize(x, norm_mean, norm_std))
                    writer.add_image(f'{phase}/images', grid, epoch)
                    grid = torchvision.utils.make_grid(pred)
                    writer.add_image(f'{phase}/prediction', grid, epoch)

            if valid_iou > best_valid_iou:
                best_valid_iou = valid_iou
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_metric': best_valid_iou
                }
                torch.save(state, os.path.join(args.weights, f'{args.model}.pth'))
    elif args.test:
        model.eval()
        iou_scores = np.array([])
        for i, (x, gt, original_shape) in enumerate(dataloaders['test']):
            x, gt = x.to(device), gt.to(device, dtype=torch.float32)
            pred = model(x)
            pred, gt = torch.sigmoid(pred.detach()), gt.detach()
            iou = metric_fn(pred, gt).item()
            iou_scores = np.append(iou_scores, iou)

        iou = iou_scores.mean()
        print(f'Test IoU: {iou:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--model', type=str, help='model name', default='deeplabv3plus')
    parser.add_argument('--epochs', type=int, metavar='INT', default=100)
    parser.add_argument('--batch_size', type=int, metavar='INT', default=12)
    parser.add_argument('--num_classes', type=int, metavar='INT', default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--size', type=int, metavar='INT', default=224, help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    parser.add_argument('--weights', type=str, default='weights', help='save weights in this directory')
    parser.add_argument('--ckpts', type=str)
    parser.add_argument('--onnx-export', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--train-val', dest='train', action='store_true')
    parser.add_argument('--no-train-val', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)
    main(parser.parse_args())
