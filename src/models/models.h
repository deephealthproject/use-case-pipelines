#include "ecvl/eddl.h"

// Model VGG16
layer VGG16(layer x, const int& num_classes);

// Model LeNet (same as https://github.com/pytorch/examples/blob/master/mnist/main.py)
layer LeNet(layer x, const int& num_classes);

// Model U-Net
layer UNet(layer x, const int& num_classes);

layer UNetWithPadding(layer x, const int& num_classes);

layer UNetWithPaddingBN(layer x, const int& num_classes);

layer SegNet(layer x, const int& num_classes);

layer SegNetBN(layer x, const int& num_classes);

layer FakeNet(layer x, const int& num_classes);