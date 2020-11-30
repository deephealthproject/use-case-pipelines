#ifndef MODELS_H_
#define MODELS_H_

#include "ecvl/support_eddl.h"

// Model LeNet (same as https://github.com/pytorch/examples/blob/master/mnist/main.py)
eddl::layer LeNet(eddl::layer x, const int& num_classes);

// Model VGG16
eddl::layer VGG16(eddl::layer x, const int& num_classes);
eddl::layer VGG16_inception_1(eddl::layer x, const int& num_classes);
eddl::layer VGG16_inception_2(eddl::layer x, const int& num_classes);

// Model SegNet (https://mi.eng.cam.ac.uk/projects/segnet)
eddl::layer SegNet(eddl::layer x, const int& num_classes);
eddl::layer SegNetBN(eddl::layer x, const int& num_classes);

// Model U-Net (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)
eddl::layer UNetWithPadding(eddl::layer x, const int& num_classes);
eddl::layer UNetWithPaddingBN(eddl::layer x, const int& num_classes);
eddl::layer UNetWithPaddingBN_v001(eddl::layer x, const int& num_classes);

// Nabla-net (https://www.hal.inserm.fr/inserm-01397806/file/MSSEG_Challenge_Proceedings.pdf 37-43)
eddl::layer Nabla(eddl::layer x, const int& num_classes);

eddl::layer ResNet_01(eddl::layer x, const int& num_classes);

#endif // MODELS_H_
