import argparse
import numpy as np
import pathlib
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from models.SegNet import SegNet
from models.PadUNet import PadUNet
from dataset import PneumothoraxDataset
from eval import Eval as Evaluator
from losses import FocalLoss2d, ComboLoss, DiceLoss


class Pneumothorax:
    def __init__(self, args):
        self.args = args
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, 'runs', args.experiment_name))
        self.current_epoch = 0
        self.best_metric = 0
        self.onnx_input = None

        if args.loss_type == 'BCE':
            self.loss = nn.BCELoss().to(self.device)
        elif args.loss_type == 'Focal':
            self.loss = FocalLoss2d().to(self.device)
        elif args.loss_type == 'Combo':
            weights = {
                'bce': 5,
                'dice': 1,
                'focal': 5
            }
            self.loss = ComboLoss(weights).to(self.device)
        elif args.loss_type == 'Dice':
            self.loss = DiceLoss().to(self.device)
        else:
            raise Exception('no valid loss function')

        if args.model == 'PadUNet':
            self.model = PadUNet(args.num_classes).to(self.device)
        elif args.model == 'SegNet':
            self.model = SegNet().to(self.device)
        else:
            raise Exception('no valid model specified')

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr)

        if args.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', verbose=True,
                                                                        threshold=0.004)

        self.evaluator = Evaluator()

        self.train_set = PneumothoraxDataset(partition='train', args=self.args)
        self.val_set = PneumothoraxDataset(partition='val', args=self.args)

        self.train_loader = data.DataLoader(self.train_set,
                                            batch_size=self.args.batch_size,
                                            shuffle=True,
                                            num_workers=self.args.data_loader_workers,
                                            pin_memory=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(self.val_set,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                          num_workers=self.args.data_loader_workers,
                                          pin_memory=True,
                                          drop_last=False)

    def train(self):
        loss_list = []
        for epoch in range(self.current_epoch, self.args.num_epochs):
            self.model.train()
            loss_list.clear()
            self.evaluator.reset_eval()

            for i, (images, labels) in enumerate(self.train_loader):
                if self.cuda:
                    images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)

                cur_loss = self.loss(outputs, labels)
                if np.isnan(cur_loss.item()):
                    raise ValueError('Loss is nan during training...')
                loss_list.append(cur_loss.item())

                cur_loss.backward()
                self.optimizer.step()

                # Track the metric
                labels = torch.squeeze(labels)
                outputs = torch.squeeze(outputs)
                outputs = outputs.data.cpu().numpy()
                labels = labels.cpu().numpy()
                self.evaluator.dice_coefficient(outputs, labels)

                if self.onnx_input is None:
                    self.onnx_input = images

            epoch_train_loss = sum(loss_list) / len(loss_list)
            epoch_train_metric = self.evaluator.mean_metric()
            self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
            self.writer.add_scalar('Metric/train', epoch_train_metric, epoch)

            print(f'Train Epoch [{epoch}/{self.args.num_epochs}], Train Mean Loss: {epoch_train_loss}, '
                  f'Train Mean Metric: {epoch_train_metric}')

            # validation
            epoch_val_metric = self.validate(epoch)

            if args.scheduler == 'plateau':
                self.scheduler.step(epoch_val_metric)

            if epoch % 10 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'metric': epoch_val_metric
                }

                torch.save(state, os.path.join(args.checkpoint_dir, args.experiment_name, 'pneumothorax_model_' +
                                               self.args.model + '_loss_' + self.args.loss_type + '_lr_' +
                                               str(self.args.lr) + '_size_' + str(self.args.resize_dims) +
                                               '_epoch_' + str(epoch) + '.pth'))
                torch.onnx.export(
                    self.model,
                    self.onnx_input,
                    os.path.join(args.checkpoint_dir, args.experiment_name, 'pneumothorax_model_' +
                                 self.args.model + '_loss_' + self.args.loss_type + '_lr_' + str(self.args.lr) +
                                 '_size_' + str(self.args.resize_dims) + '_epoch_' + str(epoch) + '.onnx'),
                    export_params=True,
                    opset_version=11
                )

    def validate(self, epoch):
        print('Validating one epoch...')
        self.model.eval()
        with torch.no_grad():
            loss_list = []
            self.evaluator.reset_eval()
            for i, (images, labels) in enumerate(self.val_loader):
                if self.cuda:
                    images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                cur_loss = self.loss(outputs, labels)
                loss_list.append(cur_loss.item())

                # Track the metric
                labels = torch.squeeze(labels)
                outputs = torch.squeeze(outputs)
                outputs = outputs.data.cpu().numpy()
                labels = labels.cpu().numpy()
                self.evaluator.dice_coefficient(outputs, labels)
                if self.onnx_input is None:
                    self.onnx_input = images

        epoch_val_loss = sum(loss_list) / len(loss_list)
        epoch_val_metric = self.evaluator.mean_metric()
        self.writer.add_scalar('Loss/validation', epoch_val_loss, epoch)
        self.writer.add_scalar('Metric/validation', epoch_val_metric, epoch)
        print(f'Validation Epoch [{epoch}/{self.args.num_epochs}], Validation Mean Loss: {epoch_val_loss}, '
              f'Validation Mean Metric: {epoch_val_metric}')

        is_best = epoch_val_metric > self.best_metric
        if is_best:
            self.best_metric = epoch_val_metric

            state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'metric': self.best_metric
            }
            torch.save(state, os.path.join(args.checkpoint_dir, args.experiment_name, 'pneumothorax_model_' +
                                           self.args.model + '_loss_' + self.args.loss_type + '_lr_' +
                                           str(self.args.lr) + '_size_' + str(self.args.resize_dims) +
                                           '_epoch_' + str(epoch) + '.pth'))
            torch.onnx.export(
                self.model,
                self.onnx_input,
                os.path.join(args.checkpoint_dir, args.experiment_name, 'pneumothorax_model_' + self.args.model +
                             '_loss_' + self.args.loss_type + '_lr_' + str(self.args.lr) + '_size_' +
                             str(self.args.resize_dims) + '_epoch_' + str(epoch) + '.onnx'),
                export_params=True,
                opset_version=11
            )

        return epoch_val_metric


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--experiment_name', default="exp1",
                            help="Name of the experiment")
    arg_parser.add_argument('--checkpoint_dir', default=None,
                            help="Directory where the checkpoints are stored")
    arg_parser.add_argument('--checkpoint_file', default=None,
                            help="Path to the onnx checkpoint file")
    arg_parser.add_argument('--dataset_filepath', default=None,
                            help="Dataset path")
    arg_parser.add_argument('--num_classes', default=1, type=int, help='Number of output classes')
    arg_parser.add_argument('--data_loader_workers', default=8, type=int, help='num_workers of Dataloader')
    arg_parser.add_argument('--num_epochs', default=100, type=int, help="Number of training epochs")
    arg_parser.add_argument('--batch_size', default=2, type=int, help='Number of images for each batch')
    arg_parser.add_argument('--resize_dims', default=512, type=int, help='Size to which resize the input images')
    arg_parser.add_argument('--loss_type', default="BCE", help='Loss function')
    arg_parser.add_argument('--scheduler', default=None, help="Scheduler used (only 'plateau' available)")
    arg_parser.add_argument('--model', default="SegNet", help='Model of the network')
    arg_parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')

    args = arg_parser.parse_args()

    p = pathlib.Path(os.path.join(args.checkpoint_dir, args.experiment_name))
    p.mkdir(parents=True, exist_ok=True)

    agent = Pneumothorax(args=args)

    if agent.cuda:
        print(f"This model will run on {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("This model will run on CPU")

    if args.checkpoint_file is not None:
        try:
            print(f"Loading checkpoint '{args.checkpoint_file}'")
            checkpoint = torch.load(args.checkpoint_file)
            agent.model.load_state_dict(checkpoint['state_dict'])
            agent.current_epoch = checkpoint['epoch'] + 1
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            agent.metric = checkpoint['metric']
            print(f"Checkpoint loaded successfully at (epoch {agent.current_epoch}), validation metric:{agent.metric})")
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(agent.args.checkpoint_dir))
            print("**First time to train**")

    agent.train()
    # agent.validate(agent.current_epoch)
