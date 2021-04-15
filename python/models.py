# Copyright (c) 2020 CRS4
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

import pyeddl.eddl as eddl

# ResNet models exported by PyTorch (using training mode)
resnet_zoo = {
    'resnet18': {
        'url': 'https://drive.google.com/uc?id=1-ZoqAQm8Ie_gHc1ozs6bbBOWVCZFvJ8x&export=download',
        'to_remove': 'Gemm_68',
        'flatten': 'Flatten_67',
        'input': 'input',
    },
    'resnet50': {
        'url': "https://drive.google.com/uc?id=1jVVVgJcImHit9xkzxpu4I9Rho4Yh2k2H&export=download",
        'to_remove': 'Gemm_174',
        'flatten': 'Flatten_173',
        'input': 'input',
    },
}

mnist_zoo = {
    'mnist': {
        'url': "https://github.com/onnx/models/blob/master/vision/classification/mnist/model/mnist-8.onnx",
        'to_remove': '',
        'flatten': '',
    }
}


def LeNet(in_layer, num_classes):
    x = in_layer
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 20, [5, 5])), [2, 2], [2, 2])
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 50, [5, 5])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 500))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.ReLu(eddl.Dense(x, 4096))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def SegNet(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3], [1, 1], "same"))
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3], [1, 1], "same"))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")
    return x


def SegNetBN(x, num_classes):
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 128, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 128, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.MaxPool(x, [2, 2], [2, 2])

    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 512, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 256, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 128, [3, 3], [1, 1], "same"), True))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 128, [3, 3], [1, 1], "same"), True))
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.UpSampling(x, [2, 2])
    x = eddl.ReLu(eddl.BatchNormalization(
        eddl.Conv(x, 64, [3, 3], [1, 1], "same"), True))
    x = eddl.Conv(x, num_classes, [3, 3], [1, 1], "same")

    return x
