#include "models.h"

using namespace eddl;

layer VGG16(layer x, const int& num_classes)
{
    x = Activation(Conv(x, 64, { 3,3 }), "relu");
    x = MaxPool(Activation(Conv(x, 64, { 3,3 }), "relu"), { 2,2 }, { 2,2 });
    x = Activation(Conv(x, 128, { 3,3 }), "relu");
    x = MaxPool(Activation(Conv(x, 128, { 3,3 }), "relu"), { 2,2 }, { 2,2 });
    x = Activation(Conv(x, 256, { 3,3 }), "relu");
    x = Activation(Conv(x, 256, { 3,3 }), "relu");
    x = MaxPool(Activation(Conv(x, 256, { 3,3 }), "relu"), { 2,2 }, { 2,2 });
    x = Activation(Conv(x, 512, { 3,3 }), "relu");
    x = Activation(Conv(x, 512, { 3,3 }), "relu");
    x = MaxPool(Activation(Conv(x, 512, { 3,3 }), "relu"), { 2,2 }, { 2,2 });
    x = Activation(Conv(x, 512, { 3,3 }), "relu");
    x = Activation(Conv(x, 512, { 3,3 }), "relu");
    x = MaxPool(Activation(Conv(x, 512, { 3,3 }), "relu"), { 2,2 }, { 2,2 });

    x = Reshape(x, { -1 });
    x = Activation(Dense(x, 4096), "relu");
    x = Activation(Dense(x, 4096), "relu");
    x = Activation(Dense(x, num_classes), "softmax");

    return x;
}

layer LeNet(layer x, const int& num_classes)
{
    x = MaxPool(Activation(Conv(x, 20, { 5,5 }), "relu"), { 2,2 }, { 2,2 });
    x = MaxPool(Activation(Conv(x, 50, { 5,5 }), "relu"), { 2,2 }, { 2,2 });
    x = Reshape(x, { -1 });
    x = Activation(Dense(x, 500), "relu");
    x = Activation(Dense(x, num_classes), "softmax");

    return x;
}