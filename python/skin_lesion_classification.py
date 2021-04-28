# Copyright (c) 2020 CRS4 Università degli Studi di Modena e Reggio Emilia (UNIMORE), AImageLab
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
UC12 Skin lesion classification pipeline.

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
from python.models import classification_zoo


def main(args):
    batch_size = args.batch_size
    image_size = args.size, args.size

    if args.weights:
        os.makedirs(args.weights, exist_ok=True)

    training_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size, ecvl.InterpolationType.cubic),
        ecvl.AugMirror(.5),
        ecvl.AugFlip(.5),
        ecvl.AugRotate([-180, 180]),
        ecvl.AugAdditivePoissonNoise([0, 10]),
        ecvl.AugGammaContrast([0.5, 1.5]),
        ecvl.AugGaussianBlur([0, 0.8]),
        ecvl.AugCoarseDropout([0, 0.03], [0.02, 0.05], 0.25),
        ecvl.AugToFloat32(255),
    ])
    validation_test_augs = ecvl.SequentialAugmentationContainer([
        ecvl.AugResizeDim(image_size), ecvl.AugToFloat32(255),
    ])
    dataset_augs = ecvl.DatasetAugmentations([training_augs, validation_test_augs, validation_test_augs])

    print('Reading dataset')
    d = ecvl.DLDataset(args.in_ds, args.batch_size, dataset_augs, ctype=ecvl.ColorType.RGB)
    num_classes = len(d.classes_)
    size = d.n_channels_, args.size, args.size

    if args.ckpts:
        net = eddl.import_net_from_onnx_file(args.ckpts, size)
    else:
        model_path = utils.DownloadModel(classification_zoo[args.model]['url'], f'{args.model}.onnx', 'model_onnx')
        net = eddl.import_net_from_onnx_file(model_path, size)
        eddl.removeLayer(net, classification_zoo[args.model]['to_remove'])  # remove last Linear of resnet
        top = eddl.getLayer(net, classification_zoo[args.model]['top'])  # get flatten of resnet

        out = eddl.Softmax(eddl.Dense(top, num_classes, True, 'classifier'))  # true is for the bias
        data_input = eddl.getLayer(net, classification_zoo[args.model]['input'])  # input of the onnx
        net = eddl.Model([data_input], [out])

    eddl.build(
        net,
        eddl.adam(args.learning_rate),
        ['softmax_cross_entropy'],
        ['accuracy'],
        eddl.CS_GPU(args.gpu, mem="low_mem") if args.gpu else eddl.CS_CPU(),
        False
    )
    out = eddl.getOut(net)[0]

    if not args.ckpts:
        eddl.initializeLayer(net, "classifier")

    eddl.summary(net)
    eddl.setlogfile(net, 'skin_lesion_classification')

    x = Tensor([batch_size, *size])
    y = Tensor([batch_size, num_classes])

    metric_fn = eddl.getMetric('accuracy')
    best_accuracy = 0.
    if args.train:
        num_samples_train = len(d.GetSplit())
        num_batches_train = num_samples_train // args.batch_size
        num_samples_val = len(d.GetSplit(ecvl.SplitType.validation))
        num_batches_val = num_samples_val // args.batch_size

        print('Starting training')
        for e in range(args.epochs):
            if args.out_dir:
                current_path = os.path.join(args.out_dir, f'Epoch_{e}')
                for c in d.classes_:
                    c_dir = os.path.join(current_path, c)
                    os.makedirs(c_dir, exist_ok=True)
            d.SetSplit(ecvl.SplitType.training)
            eddl.reset_loss(net)
            s = d.GetSplit()
            random.shuffle(s)
            d.split_.training_ = s
            d.ResetAllBatches()
            for b in range(num_batches_train):
                d.LoadBatch(x, y)
                eddl.train_batch(net, [x], [y])
                losses = eddl.get_losses(net)
                metrics = eddl.get_metrics(net)

                print(f'Train - epoch [{e + 1}/{args.epochs}] - batch [{b + 1}/{num_batches_train}]'
                      f' - loss={losses[0]:.3f} - accuracy={metrics[0]:.3f}', flush=True)

            d.SetSplit(ecvl.SplitType.validation)
            values = np.zeros(num_batches_val)
            eddl.reset_loss(net)

            for b in range(num_batches_val):
                n = 0
                d.LoadBatch(x, y)
                eddl.forward(net, [x])
                output = eddl.getOutput(out)
                value = metric_fn.value(y, output)
                values[b] = value
                if args.out_dir:
                    for k in range(args.batch_size):
                        result = output.select([str(k)])
                        target = y.select([str(k)])
                        result_a = np.array(result, copy=False)
                        target_a = np.array(target, copy=False)
                        classe = np.argmax(result_a).item()
                        gt_class = np.argmax(target_a).item()
                        single_image = x.select([str(k)])
                        img_t = ecvl.TensorToView(single_image)
                        img_t.colortype_ = ecvl.ColorType.BGR
                        single_image.mult_(255.)
                        filename = d.samples_[d.GetSplit()[n]].location_[0]
                        head, tail = os.path.splitext(os.path.basename(filename))
                        bname = '{}_gt_class_{}.png'.format(head, gt_class)
                        cur_path = os.path.join(current_path, d.classes_[classe], bname)
                        ecvl.ImWrite(cur_path, img_t)
                    n += 1

                print(f'Validation - epoch [{e + 1}/{args.epochs}] - batch [{b + 1}/{num_batches_val}] -'
                      f' accuracy={np.mean(values[:b + 1] / batch_size):.3f}')

            last_accuracy = np.mean(values / batch_size)
            print(f'Validation - epoch [{e + 1}/{args.epochs}] - total accuracy={last_accuracy:.3f}')
            if last_accuracy > best_accuracy:
                best_accuracy = last_accuracy
                print('Saving weights')
                eddl.save_net_to_onnx_file(net, f'isic_classification_{args.model}_epoch_{e + 1}.onnx')

    elif args.test:
        d.SetSplit(ecvl.SplitType.test)
        num_samples_test = len(d.GetSplit())
        num_batches_test = num_samples_test // batch_size
        values = np.zeros(num_batches_test)
        eddl.reset_loss(net)

        for b in range(num_batches_test):
            d.LoadBatch(x, y)
            eddl.forward(net, [x])
            output = eddl.getOutput(out)
            value = metric_fn.value(y, output)
            values[b] = value
            if args.out_dir:
                n = 0
                for k in range(args.batch_size):
                    result = output.select([str(k)])
                    target = y.select([str(k)])
                    result_a = np.array(result, copy=False)
                    target_a = np.array(target, copy=False)
                    classe = np.argmax(result_a).item()
                    gt_class = np.argmax(target_a).item()
                    single_image = x.select([str(k)])
                    img_t = ecvl.TensorToView(single_image)
                    img_t.colortype_ = ecvl.ColorType.BGR
                    single_image.mult_(255.)
                    filename = d.samples_[d.GetSplit()[n]].location_[0]
                    head, tail = os.path.splitext(os.path.basename(filename))
                    bname = "%s_gt_class_%s.png" % (head, gt_class)
                    cur_path = os.path.join(args.out_dir, d.classes_[classe], bname)
                    ecvl.ImWrite(cur_path, img_t)
                    n += 1

            print(f'Test - batch [{b + 1}/{num_batches_test}] - accuracy={np.mean(values[:b + 1] / batch_size):.3f}')
        print(f'Test - total accuracy={np.mean(values / batch_size):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('in_ds', metavar='INPUT_DATASET')
    parser.add_argument('--ckpts', type=str, help='Load an existing ONNX')
    parser.add_argument('--model', type=str, help='Model to use for training from scratch', default='resnet50')
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
