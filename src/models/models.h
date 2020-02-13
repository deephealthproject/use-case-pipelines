#ifndef MODELS_H_
#define MODELS_H_

#include "ecvl/support_eddl.h"

// Model LeNet (same as https://github.com/pytorch/examples/blob/master/mnist/main.py)
layer LeNet(layer x, const int& num_classes);

// Model VGG16
layer VGG16(layer x, const int& num_classes);

// Model SegNet (https://mi.eng.cam.ac.uk/projects/segnet)
layer SegNet(layer x, const int& num_classes);
layer SegNetBN(layer x, const int& num_classes);

// Model U-Net (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)
layer UNetWithPadding(layer x, const int& num_classes);
layer UNetWithPaddingBN(layer x, const int& num_classes);

#endif // MODELS_H_