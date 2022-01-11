# Copyright (c) 2021 CRS4 Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
UC12 Skin lesion Segmentation pipeline.

It loads a ResNet model already pretrained on ImageNet and the fine-tunes it on skin lesion data.
"""

import argparse
import os
import random

import numpy as np
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor

from python import utils
from python.models import Unet


def main(args):
    batch_size = args.batch_size
    image_size = args.size, args.size
    thresh = 0.5

    if args.weights:
        os.makedirs(args.weights, exist_ok=True)

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size, ecvl.InterpolationType.cubic, gt_interp=ecvl.InterpolationType.nearest),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180]),
        ecvl.AugAdditivePoissonNoise([0, 10]),
        ecvl.AugGammaContrast([0.5, 1.5]),
        ecvl.AugGaussianBlur([0, 0.8]),
        ecvl.AugCoarseDropout([0, 0.03], [0.02, 0.05], 0.25),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize([0.6681, 0.5301, 0.5247], [0.1337, 0.1480, 0.1595]),  # isic stats

    ])
    validation_test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size, ecvl.InterpolationType.cubic, gt_interp=ecvl.InterpolationType.nearest),
        ecvl.AugToFloat32(255, divisor_gt=255),
        ecvl.AugNormalize([0.6681, 0.5301, 0.5247], [0.1337, 0.1480, 0.1595]),  # isic stats

    ])
    dataset_augs = ecvl.DatasetAugmentations([training_augs, validation_test_augs, validation_test_augs])

    print('Reading dataset')
    d = ecvl.DLDataset(args.in_ds, args.batch_size, dataset_augs, ctype=ecvl.ColorType.RGB)
    num_classes = len(d.classes_) or d.n_channels_gt_
    size = d.n_channels_, args.size, args.size

    if args.ckpts:
        net = eddl.import_net_from_onnx_file(args.ckpts, size)
    else:
        in_ = eddl.Input(size)
        out = Unet(in_, num_classes)
        out_sigm = eddl.Sigmoid(out)
        net = eddl.Model([in_], [out_sigm])

        # model_path = utils.DownloadModel(segmentation_zoo[args.model]['url'], f'{args.model}.onnx', 'model_onnx')
        # net = eddl.import_net_from_onnx_file(model_path, size)
        # eddl.removeLayer(net, segmentation_zoo[args.model]['to_remove'])
        # top = eddl.getLayer(net, segmentation_zoo[args.model]['top'])
        #
        # out = eddl.Sigmoid(eddl.Conv(top, num_classes, [3, 3], name='last_layer'))
        # data_input = eddl.getLayer(net, segmentation_zoo[args.model]['input'])  # input of the onnx
        # net = eddl.Model([data_input], [out])

    loss_name = 'binary_cross_entropy'
    metric_name = 'mean_squared_error'
    eddl.build(
        net,
        eddl.adam(args.learning_rate),
        [loss_name],
        [metric_name],
        eddl.CS_GPU(args.gpu, mem="low_mem") if args.gpu else eddl.CS_CPU(),
        True
    )
    out = eddl.getOut(net)[0]

    # if not args.ckpts:
    #     eddl.initializeLayer(net, "last_layer")

    eddl.summary(net)
    eddl.setlogfile(net, 'skin_lesion_segmentation')

    x = Tensor([args.batch_size, *size])
    y = Tensor([args.batch_size, d.n_channels_gt_, size[1], size[2]])

    miou = 0.
    if args.train:
        num_samples_train = len(d.GetSplit())
        num_batches_train = num_samples_train // args.batch_size
        num_samples_val = len(d.GetSplit(ecvl.SplitType.validation))
        num_batches_val = num_samples_val // args.batch_size
        evaluator = utils.Evaluator()

        print('Starting training')
        for e in range(args.epochs):
            d.SetSplit(ecvl.SplitType.training)
            eddl.reset_loss(net)
            s = d.GetSplit()
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetAllBatches()
            for b in range(num_batches_train):
                d.LoadBatch(x, y)
                # x_ = x.select(["0"])
                # x_.normalize_(0, 1)
                # x_.mult_(255.)
                # x_.save(f'images/train_{e}_{b}.png')
                #
                # y_ = y.select(["0"])
                # # y_.mult_(255.)
                # y_.save(f'images/train_gt_{e}_{b}.png')

                eddl.train_batch(net, [x], [y])
                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                print(f'Train - epoch [{e + 1}/{args.epochs}] - batch [{b + 1}/{num_batches_train}]'
                      f' - {loss_name}={losses[0]:.3f} - {metric_name}={metrics[0]:.3f}', flush=True)

            d.SetSplit(ecvl.SplitType.validation)
            evaluator.ResetEval()
            eddl.reset_loss(net)

            for b in range(num_batches_val):
                n = 0
                print(f'Validation - epoch [{e + 1}/{args.epochs}] - batch [{b + 1}/{num_batches_val}]')
                d.LoadBatch(x, y)
                eddl.forward(net, [x])
                output = eddl.getOutput(out)
                for bs in range(args.batch_size):
                    img = output.select([str(bs)])
                    gt = y.select([str(bs)])
                    img_np = np.array(img, copy=False)
                    gt_np = np.array(gt, copy=False)
                    iou = evaluator.BinaryIoU(img_np, gt_np, thresh=thresh)
                    print(f' - IoU: {iou:.3f}', end="", flush=True)
                    if args.out_dir:
                        # C++ BinaryIoU modifies image as a side effect
                        img_np[img_np >= thresh] = 1
                        img_np[img_np < thresh] = 0
                        img_t = ecvl.TensorToView(img)
                        img_t.colortype_ = ecvl.ColorType.GRAY
                        img_t.channels_ = "xyc"
                        img.mult_(255.)
                        # orig_img
                        orig_img = x.select([str(bs)])
                        orig_img.mult_(255.)
                        orig_img_t = ecvl.TensorToImage(orig_img)
                        orig_img_t.colortype_ = ecvl.ColorType.BGR
                        orig_img_t.channels_ = "xyc"

                        tmp, labels = ecvl.Image.empty(), ecvl.Image.empty()
                        ecvl.CopyImage(img_t, tmp, ecvl.DataType.uint8)
                        ecvl.ConnectedComponentsLabeling(tmp, labels)
                        ecvl.CopyImage(labels, tmp, ecvl.DataType.uint8)
                        contours = ecvl.FindContours(tmp)
                        ecvl.CopyImage(orig_img_t, tmp, ecvl.DataType.uint8)
                        tmp_np = np.array(tmp, copy=False)
                        for cseq in contours:
                            for c in cseq:
                                tmp_np[c[0], c[1], 0] = 0
                                tmp_np[c[0], c[1], 1] = 0
                                tmp_np[c[0], c[1], 2] = 255
                        filename = d.samples_[d.GetSplit()[n]].location_[0]
                        head, tail = os.path.splitext(os.path.basename(filename))
                        bname = "%s.png" % head
                        output_fn = os.path.join(args.out_dir, bname)
                        ecvl.ImWrite(output_fn, tmp)
                        if e == 0:
                            gt_t = ecvl.TensorToView(gt)
                            gt_t.colortype_ = ecvl.ColorType.GRAY
                            gt_t.channels_ = "xyc"
                            gt.mult_(255.)
                            gt_filename = d.samples_[d.GetSplit()[n]].label_path_
                            gt_fn = os.path.join(
                                args.out_dir, os.path.basename(gt_filename)
                            )
                            ecvl.ImWrite(gt_fn, gt_t)
                    n += 1
                print()

            last_miou = evaluator.MIoU()
            print(f'Validation - epoch [{e + 1}/{args.epochs}] - Total MIoU: {last_miou:.3f}')

            if last_miou > miou:
                miou = last_miou
                eddl.save_net_to_onnx_file(net,
                                           os.path.join(args.weights, f'isic_segm_{args.model}_epoch_{e + 1}.onnx'))
                print('Weights saved')
    elif args.test:
        evaluator = utils.Evaluator()
        evaluator.ResetEval()

        d.SetSplit(ecvl.SplitType.test)
        num_samples_test = len(d.GetSplit())
        num_batches_test = num_samples_test // batch_size
        for b in range(num_batches_test):
            n = 0
            print(f'Test - batch [{b + 1}/{num_batches_test}]')
            d.LoadBatch(x, y)
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            for bs in range(args.batch_size):
                img = output.select([str(bs)])
                gt = y.select([str(bs)])
                img_np, gt_np = np.array(img, copy=False), np.array(gt, copy=False)
                iou = evaluator.BinaryIoU(img_np, gt_np, thresh=thresh)
                print(f' - IoU: {iou:.3f}', end="", flush=True)
                if args.out_dir:
                    # C++ BinaryIoU modifies image as a side effect
                    img_np[img_np >= thresh] = 1
                    img_np[img_np < thresh] = 0
                    img_t = ecvl.TensorToView(img)
                    img_t.colortype_ = ecvl.ColorType.GRAY
                    img_t.channels_ = "xyc"
                    img.mult_(255.)
                    # orig_img
                    orig_img = x.select([str(bs)])
                    orig_img.mult_(255.)
                    orig_img_t = ecvl.TensorToImage(orig_img)
                    orig_img_t.colortype_ = ecvl.ColorType.BGR
                    orig_img_t.channels_ = "xyc"

                    tmp, labels = ecvl.Image.empty(), ecvl.Image.empty()
                    ecvl.CopyImage(img_t, tmp, ecvl.DataType.uint8)
                    ecvl.ConnectedComponentsLabeling(tmp, labels)
                    ecvl.CopyImage(labels, tmp, ecvl.DataType.uint8)
                    contours = ecvl.FindContours(tmp)
                    ecvl.CopyImage(orig_img_t, tmp, ecvl.DataType.uint8)
                    tmp_np = np.array(tmp, copy=False)
                    for cseq in contours:
                        for c in cseq:
                            tmp_np[c[0], c[1], 0] = 0
                            tmp_np[c[0], c[1], 1] = 0
                            tmp_np[c[0], c[1], 2] = 255
                    filename = d.samples_[d.GetSplit()[n]].location_[0]
                    head, tail = os.path.splitext(os.path.basename(filename))
                    bname = "%s.png" % head
                    output_fn = os.path.join(args.out_dir, bname)
                    ecvl.ImWrite(output_fn, tmp)

                    gt_t = ecvl.TensorToView(gt)
                    gt_t.colortype_ = ecvl.ColorType.GRAY
                    gt_t.channels_ = "xyc"
                    gt.mult_(255.)
                    gt_filename = d.samples_[d.GetSplit()[n]].label_path_
                    gt_fn = os.path.join(
                        args.out_dir, os.path.basename(gt_filename)
                    )
                    ecvl.ImWrite(gt_fn, gt_t)
                n += 1
        miou = evaluator.MIoU()
        print(f'Test - Total MIoU: {miou:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--ckpts', type=str, help='Load an existing ONNX')
    parser.add_argument('--model', type=str, help='Model to use for training from scratch', default='Unet')
    parser.add_argument('--epochs', type=int, metavar='INT', default=30)
    parser.add_argument('--batch-size', type=int, metavar='INT', default=24)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--size', type=int, metavar='INT', default=224, help='Size of input slices')
    parser.add_argument('--gpu', nargs='+', type=int, required=False, help='`--gpu 1 1` to use two GPUs')
    parser.add_argument('--out-dir', metavar='DIR', help='if set, save images in this directory')
    parser.add_argument('--weights', metavar='DIR', type=str, default='weights', help='save weights in this directory')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)
    parser.add_argument('--train-val', dest='train', action='store_true')
    parser.add_argument('--no-train-val', dest='train', action='store_false')
    parser.set_defaults(train=True)
    main(parser.parse_args())
