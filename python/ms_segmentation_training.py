# Copyright (c) 2020, Università degli Studi di Modena e Reggio Emilia
# (UNIMORE), AImageLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""\
MS segmentation training example.
"""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import pyecvl.ecvl as ecvl

import utils
import models


class MSVolume:
    def __init__(self, dataset, n_channels, stride=None):
        self.d_ = dataset
        self.n_channels_ = n_channels
        self.stride_ = n_channels if stride is None else stride

        self.slices_ = 0
        self.current_slice_ = 0
        self.current_volume_ = -1
        self.indices_ = []
        self.names_ = []

        self.volume_ = ecvl.Image.empty()
        self.gt_ = ecvl.Image.empty()

    def Reset(self):
        self.current_volume_ = -1
        self.current_slice_ = 0
        self.indices_ = []
        self.d_.ResetAllBatches()

    def Init(self):
        self.current_volume_ += 1
        # Get next volume from DLDataset
        index = self.d_.GetSplit()[self.current_volume_]
        elem = self.d_.samples_[index]

        # Load a volume and its gt in memory
        self.volume_ = elem.LoadImage(self.d_.ctype_, False)
        tmp = elem.LoadImage(self.d_.ctype_gt_, True)
        ecvl.CopyImage(tmp, self.gt_, ecvl.DataType.float32)

        self.current_slice_ = 0
        self.slices_ = self.volume_.Channels()
        # indices created as random between 0 and slices_ / stride_
        self.indices_ = np.arange(0, self.slices_ // self.stride_)
        np.random.shuffle(self.indices_)

        # Save names of current volume and gt --> for save_images
        self.names_.clear()
        sample_name = Path(elem.location_[0]).parent.stem + '_' + Path(elem.location_[0]).stem + '_'
        self.names_.append(sample_name)
        sample_name = Path(elem.label_path_).parent.stem + '_' + Path(elem.label_path_).stem + '_'
        self.names_.append(sample_name)

    def LoadBatch(self):
        bs = self.d_.batch_size_

        if self.d_.current_split_ == ecvl.SplitType.training:
            index = 0
        elif self.d_.current_split_ == ecvl.SplitType.validation:
            index = 1
        else:
            index = 2

        start = self.d_.current_batch_[index] * bs
        # Load a new volume if we already loaded all slices of the current one
        if self.current_slice_ >= len(self.indices_) or len(self.indices_) < start + bs:
            # Stop training/validation if there are no more volume to read
            if self.current_volume_ >= len(self.d_.GetSplit()) - 1:
                return False, None, None
            # Load a new volume
            self.Init()
            self.d_.ResetCurrentBatch()

        # Because of https://github.com/pybind/pybind11/issues/175
        self.d_.current_batch_ = [elem + 1 if i == index else elem for i, elem in
                                  enumerate(self.d_.current_batch_)]

        images = []
        labels = []
        # Fill tensors with data
        for i in range(start, start + bs):
            assert self.current_slice_ < len(self.indices_)
            # Read slices and their ground truth
            depth_start = self.indices_[self.current_slice_] * self.stride_
            v_volume_ = np.array(self.volume_, copy=False)[:, :, depth_start:depth_start + self.n_channels_]
            v_volume_ = ecvl.Image.fromarray(v_volume_, self.volume_.channels_, self.volume_.colortype_)
            v_gt_ = np.array(self.gt_, copy=False)[:, :, depth_start:depth_start + self.n_channels_]
            v_gt_ = ecvl.Image.fromarray(v_gt_, self.volume_.channels_, self.volume_.colortype_)

            # Apply chain of augmentations to sample image and corresponding ground truth
            self.d_.augs_.Apply(self.d_.current_split_, v_volume_, v_gt_)

            # Copy image into tensor (images)
            images.append(ecvl.ImageToTensor(v_volume_))
            # Copy label into tensor (labels)
            labels.append(ecvl.ImageToTensor(v_gt_))

            self.current_slice_ += 1

        return True, images, labels


def fill_tensors(images, labels, x, y):
    x_np = np.array(x, copy=False)
    y_np = np.array(y, copy=False)

    for i, (img, lab) in enumerate(zip(images, labels)):
        img_np = np.array(img, copy=False)
        lab_np = np.array(lab, copy=False)
        x_np[i, ...] = img_np
        y_np[i, ...] = lab_np

    return x, y


def main(args):
    num_classes = args.num_classes
    size = [args.size, args.size]  # size of images
    thresh = 0.5
    best_dice = 0.
    random_weights = True

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.ckpts and os.path.exists(args.ckpts):
        print('Loading ONNX model `{}`'.format(args.ckpts))
        net = eddl.import_net_from_onnx_file(args.ckpts)
        random_weights = False
    else:
        in_ = eddl.Input([args.n_channels, size[0], size[1]])
        out = models.Nabla(in_, num_classes)
        net = eddl.Model([in_], [out])

    eddl.build(
        net,
        eddl.adam(args.learning_rate),
        ['mse'],
        ['dice'],
        eddl.CS_GPU(args.gpu, mem='low_mem') if args.gpu else eddl.CS_CPU(),
        random_weights
    )
    eddl.summary(net)
    # eddl.plot(net, 'ms_segmentation.pdf')
    # eddl.setlogfile(net, 'ms_segmentation_training')

    training_augs = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    validation_augs = ecvl.SequentialAugmentationContainer([ecvl.AugResizeDim(size)])
    dataset_augs = ecvl.DatasetAugmentations([training_augs, validation_augs, None])

    print('Reading dataset')
    d = ecvl.DLDataset(args.in_ds, args.batch_size, dataset_augs, ecvl.ColorType.none, ecvl.ColorType.none)
    v = MSVolume(d, args.n_channels)  # MSVolume takes a reference to DLDataset

    # Prepare tensors which store batches
    x = Tensor([args.batch_size, args.n_channels, size[0], size[1]])
    y = Tensor([args.batch_size, args.n_channels, size[0], size[1]])

    indices = list(range(args.batch_size))

    evaluator = utils.Evaluator()
    print('Starting training')
    for e in range(args.epochs):
        v.Reset()

        current_path = None
        if args.out_dir:
            current_path = Path(args.out_dir) / Path('Epoch_{}'.format(e))
            os.makedirs(current_path, exist_ok=True)
        d.SetSplit(ecvl.SplitType.training)

        # Reset errors
        eddl.reset_loss(net)
        # Shuffle training list
        s = d.GetSplit()
        random.shuffle(s)
        d.split_.training_ = s

        j = 0
        old_volume = 0

        while True:
            cond, images, labels = v.LoadBatch()
            if not cond:
                break  # All volumes have been processed
            x, y = fill_tensors(images, labels, x, y)

            if old_volume != v.current_volume_:
                j = 0  # Current volume ended
                old_volume = v.current_volume_

            print('Epoch {:}/{:} - volume {:}/{:} - batch {:}/{:}'.format(
                e, args.epochs - 1, v.current_volume_, len(d.GetSplit()) - 1, j,
                   v.slices_ // (v.n_channels_ * d.batch_size_) - 1), flush=True, end=' ')

            # Train batch
            eddl.train_batch(net, [x], [y], indices)
            eddl.print_loss(net, j)
            print()
            j += 1

        print('Starting validation:')
        d.SetSplit(ecvl.SplitType.validation)
        evaluator.ResetEval()
        v.Reset()

        j = 0
        old_volume = 0

        while True:
            cond, images, labels = v.LoadBatch()
            if not cond:
                break  # All volumes have been processed
            x, y = fill_tensors(images, labels, x, y)

            if old_volume != v.current_volume_:
                j = 0  # Current volume ended
                old_volume = v.current_volume_

            print('Validation - Epoch {:}/{:} - volume {:}/{:} - batch {:}/{:}'.format(
                e, args.epochs - 1, v.current_volume_, len(d.GetSplit()) - 1, j,
                   v.slices_ // (v.n_channels_ * d.batch_size_) - 1), flush=True, end='')

            eddl.forward(net, [x])
            out_t = eddl.getOut(net)
            output = eddl.getOutput(out_t[0])
            # Compute Dice metric and optionally save the output images
            for k in range(args.batch_size):
                pred = output.select([str(k)])
                gt = y.select([str(k)])
                pred_np = np.array(pred, copy=False).squeeze(0)
                gt_np = np.array(gt, copy=False).squeeze(0)

                for im in range(args.n_channels):
                    p = pred_np[im, ...]
                    g = gt_np[im, ...]
                    # DiceCoefficient modifies image as a side effect
                    dice = evaluator.DiceCoefficient(p, g, thresh=thresh)
                    print('- Dice: {:.6f} '.format(dice), end='', flush=True)

                    if args.out_dir:
                        p *= 255
                        pred_ecvl = ecvl.TensorToImage(pred)
                        pred_ecvl.colortype_ = ecvl.ColorType.GRAY
                        pred_ecvl.channels_ = 'xyc'
                        ecvl.ImWrite(str(current_path / Path(v.names_[0] + str(
                            v.indices_[v.current_slice_ - args.batch_size + k] * v.stride_ + im) + '.png')), pred_ecvl)
                        g *= 255
                        gt_ecvl = ecvl.TensorToImage(gt)
                        gt_ecvl.colortype_ = ecvl.ColorType.GRAY
                        gt_ecvl.channels_ = 'xyc'
                        ecvl.ImWrite(str(current_path / Path(v.names_[1] + str(
                            v.indices_[v.current_slice_ - args.batch_size + k] * v.stride_ + im) + '.png')), gt_ecvl)

            j += 1
            print()

        mean_dice = evaluator.MeanMetric()
        if mean_dice > best_dice:
            print('Saving ONNX')
            eddl.save_net_to_onnx_file(net, 'ms_segmentation_checkpoint_epoch_{}.onnx'.format(e))
            best_dice = mean_dice
        print('----------------------------')
        print('Mean Dice Coefficient: {:.6g}'.format(mean_dice))
        print('----------------------------')

        # Save metric values on file
        with open('output_evaluate_ms_segmentation.txt', 'a') as f:
            f.write('Epoch {} - Mean Dice Coefficient: {}'.format(e, evaluator.MeanMetric()))


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
