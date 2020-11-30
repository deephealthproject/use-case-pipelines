#include "models.h"

using namespace eddl;

layer LeNet(layer x, const int& num_classes)
{
    x = MaxPool(ReLu(Conv(x, 20, { 5,5 })), { 2,2 }, { 2,2 });
    x = MaxPool(ReLu(Conv(x, 50, { 5,5 })), { 2,2 }, { 2,2 });
    x = Reshape(x, { -1 });
    x = ReLu(Dense(x, 500));
    x = Softmax(Dense(x, num_classes));

    return x;
}

layer Nabla(layer x, const int& num_classes)
{
    layer x1, x2;
    // encoder
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x1 = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = MaxPool(x1, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));

    // decoder
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x = UpSampling(x, { 2,2 }); // should be unpooling
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x = UpSampling(x, { 2,2 }); // should be unpooling
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x = BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same"));
    x2 = UpSampling(x, { 2,2 }); // should be unpooling

    // merge
    x = Concat({ x1, x2 });

    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1,1 }, "same")));
    x = Conv(x, num_classes, { 1,1 });
    x = Sigmoid(x);

    return x;
}

layer VGG16(layer x, const int& num_classes)
{
    x = ReLu(Conv(x, 64, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 64, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 128, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }));
    x = ReLu(Conv(x, 256, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 256, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 512, { 3,3 })), { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = ReLu(Conv(x, 512, { 3,3 }));
    x = MaxPool(ReLu(Conv(x, 512, { 3,3 })), { 2,2 }, { 2,2 });

    x = Reshape(x, { -1 });
    x = ReLu(Dense(x, 4096));
    x = ReLu(Dense(x, 4096));
    x = Softmax(Dense(x, num_classes));

    return x;
}

layer VGG16_inception_1(layer x, const int& num_classes)
{
    layer l1 = ReLu(BatchNormalization(Conv(x, 20, { 3,3 }, { 1,1 }, "same")));
    layer l2 = ReLu(BatchNormalization(Conv(x, 24, { 5,5 }, { 1,1 }, "same")));
    layer l3 = ReLu(BatchNormalization(Conv(x, 30, { 7,7 }, { 1,1 }, "same")));

    layer l = Concat({ l1,l2,l3 });

    l = ReLu(BatchNormalization(Conv(l, 64, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 64, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 128, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 128, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = Reshape(l, { -1 });
    l = ReLu(BatchNormalization(Dense(l, 2048)));
    l = ReLu(BatchNormalization(Dense(l, 1024)));
    l = Softmax(Dense(l, num_classes));

    return l;
}
layer VGG16_inception_2(layer x, const int& num_classes)
{
    layer l1 = ReLu(BatchNormalization(Conv(x, 32, { 3,3 }, { 1,1 }, "same")));
    layer l2 = ReLu(BatchNormalization(Conv(x, 32, { 5,5 }, { 1,1 }, "same")));
    layer l = Concat({ l1,l2 });

    l = ReLu(BatchNormalization(Conv(l, 64, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 64, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 128, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 128, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 256, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = ReLu(BatchNormalization(Conv(l, 512, { 3,3 }, { 1,1 }, "same")));
    l = MaxPool(l, { 2,2 }, { 2,2 });

    l = Reshape(l, { -1 });
    l = ReLu(BatchNormalization(Dense(l, 4096)));
    l = ReLu(BatchNormalization(Dense(l, 2048)));
    l = Softmax(Dense(l, num_classes));

    return l;
}

layer SegNet(layer x, const int& num_classes)
{
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });

    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = Conv(x, num_classes, { 3,3 }, { 1,1 }, "same");
    x = Sigmoid(x);

    return x;
}

layer SegNetBN(layer x, const int& num_classes)
{
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 128, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 128, { 3,3 }, { 1, 1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = MaxPool(x, { 2,2 }, { 2,2 });

    x = UpSampling(x, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = UpSampling(x, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 512, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = UpSampling(x, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 256, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 128, { 3,3 }, { 1, 1 }, "same")));
    x = UpSampling(x, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 128, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = UpSampling(x, { 2,2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = Conv(x, num_classes, { 3,3 }, { 1,1 }, "same");
    x = Sigmoid(x);

    return x;
}

layer UNetWithPadding(layer x, const int& num_classes)
{
    layer x2;
    layer x3;
    layer x4;
    layer x5;

    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x2 = MaxPool(x, { 2,2 }, { 2,2 });
    x2 = ReLu(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"));
    x2 = ReLu(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"));
    x3 = MaxPool(x2, { 2,2 }, { 2,2 });
    x3 = ReLu(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"));
    x3 = ReLu(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"));
    x4 = MaxPool(x3, { 2,2 }, { 2,2 });
    x4 = ReLu(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"));
    x4 = ReLu(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"));
    x5 = MaxPool(x4, { 2,2 }, { 2,2 });
    x5 = ReLu(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same"));
    x5 = ReLu(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same"));
    x5 = Conv(UpSampling(x5, { 2,2 }), 512, { 2,2 }, { 1, 1 }, "same");

    x4 = Concat({ x4, x5 });
    x4 = ReLu(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"));
    x4 = ReLu(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"));
    x4 = Conv(UpSampling(x4, { 2,2 }), 256, { 2,2 }, { 1, 1 }, "same");

    x3 = Concat({ x3, x4 });
    x3 = ReLu(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"));
    x3 = ReLu(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"));
    x3 = Conv(UpSampling(x3, { 2,2 }), 128, { 2,2 }, { 1, 1 }, "same");

    x2 = Concat({ x2, x3 });
    x2 = ReLu(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"));
    x2 = ReLu(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"));
    x2 = Conv(UpSampling(x2, { 2,2 }), 64, { 2,2 }, { 1, 1 }, "same");

    x = Concat({ x, x2 });
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = Conv(x, num_classes, { 1,1 });
    x = Sigmoid(x);

    return x;
}

layer UNetWithPaddingBN(layer x, const int& num_classes)
{
    layer x2;
    layer x3;
    layer x4;
    layer x5;

    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x2 = MaxPool(x, { 2,2 }, { 2,2 });
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x3 = MaxPool(x2, { 2,2 }, { 2,2 });
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x4 = MaxPool(x3, { 2,2 }, { 2,2 });
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x5 = MaxPool(x4, { 2,2 }, { 2,2 });
    x5 = ReLu(BatchNormalization(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same")));
    x5 = ReLu(BatchNormalization(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same")));
    x5 = BatchNormalization(Conv(UpSampling(x5, { 2,2 }), 512, { 2,2 }, { 1, 1 }, "same"));

    x4 = Concat({ x4, x5 });
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x4 = BatchNormalization(Conv(UpSampling(x4, { 2,2 }), 256, { 2,2 }, { 1, 1 }, "same"));

    x3 = Concat({ x3, x4 });
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x3 = BatchNormalization(Conv(UpSampling(x3, { 2,2 }), 128, { 2,2 }, { 1, 1 }, "same"));

    x2 = Concat({ x2, x3 });
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x2 = BatchNormalization(Conv(UpSampling(x2, { 2,2 }), 64, { 2,2 }, { 1, 1 }, "same"));

    x = Concat({ x, x2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = Conv(x, num_classes, { 1,1 });
    x = Sigmoid(x);

    return x;
}

layer UNetWithPaddingBN_v001(layer x, const int& num_classes)
{
    layer x1_3, x1_5, x1_7;
    layer x1;
    layer x2;
    layer x3;
    layer x4;
    layer x5;
    layer y;

    x1_3 = ReLu(BatchNormalization(Conv(x, 32, {3,3}, {1,1}, "same")));
    x1_5 = ReLu(BatchNormalization(Conv(x, 32, {5,5}, {1,1}, "same")));
    x1_7 = ReLu(BatchNormalization(Conv(x, 32, {7,7}, {1,1}, "same")));

    x1 = Concat({x1_3, x1_5, x1_7});

    x1 = ReLu(BatchNormalization(Conv(x1, 96, {3,3}, {1,1}, "same")));

    x2 = MaxPool(x1, {2,2}, {2,2});
    x2 = ReLu(BatchNormalization(Conv(x2, 96, {3,3}, {1,1}, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 96, {3,3}, {1,1}, "same")));

    x3 = MaxPool(x2, {2,2}, {2,2});
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));

    x4 = MaxPool(x3, {2,2}, {2,2});
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));

    x5 = MaxPool(x4, {2,2}, {2,2});
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    x5 = BatchNormalization(Conv(UpSampling(x5, {2,2}), 256, {2,2}, {1, 1}, "same"));

    x4 = Concat({x4, x5});
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = BatchNormalization(Conv(UpSampling(x4, {2,2}), 128, {2,2}, {1,1}, "same"));

    x3 = Concat({x3,x4});
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = BatchNormalization(Conv(UpSampling(x3, {2,2}), 96, {2,2}, {1, 1}, "same"));

    x2 = Concat({x2,x3});
    x2 = ReLu(BatchNormalization(Conv(x2, 96, { 3,3 }, { 1, 1 }, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 96, { 3,3 }, { 1, 1 }, "same")));
    x2 = BatchNormalization(Conv(UpSampling(x2, {2,2}), 96, {2,2}, {1,1}, "same"));

    x1 = Concat({x1,x2});
    x1 = ReLu(BatchNormalization(Conv(x1, 64, {3,3}, {1,1}, "same")));
    x1 = ReLu(BatchNormalization(Conv(x1, 64, {3,3}, {1,1}, "same")));
    y = Conv(x1, num_classes, {1,1});
    y = Sigmoid(y);

    return y;
}

layer UNetWithPaddingBN_v002(layer x, const int& num_classes)
{
    layer x1_3, x1_5, x1_7;
    layer x1;
    layer x2;
    layer x3;
    layer x4;
    layer x5;
    layer y;

    x1_3 = ReLu(BatchNormalization(Conv(x, 32, {3,3}, {1,1}, "same")));
    x1_5 = ReLu(BatchNormalization(Conv(x, 32, {5,5}, {1,1}, "same")));
    x1_7 = ReLu(BatchNormalization(Conv(x, 32, {7,7}, {1,1}, "same")));

    x1 = Concat({x1_3, x1_5, x1_7});

    x1 = ReLu(BatchNormalization(Conv(x1, 96, {3,3}, {1,1}, "same")));

    x2 = MaxPool(x1, {2,2}, {2,2});
    x2 = ReLu(BatchNormalization(Conv(x2, 96, {3,3}, {1,1}, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 96, {3,3}, {1,1}, "same")));

    x3 = MaxPool(x2, {2,2}, {2,2});
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));

    x4 = MaxPool(x3, {2,2}, {2,2});
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));

    x5 = MaxPool(x4, {2,2}, {2,2});
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    x5 = BatchNormalization(Conv(UpSampling(x5, {2,2}), 256, {2,2}, {1, 1}, "same"));

    x4 = Concat({x4, x5});
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = BatchNormalization(Conv(UpSampling(x4, {2,2}), 128, {2,2}, {1,1}, "same"));

    x3 = Concat({x3,x4});
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = BatchNormalization(Conv(UpSampling(x3, {2,2}), 96, {2,2}, {1, 1}, "same"));

    x2 = Concat({x2,x3});
    x2 = ReLu(BatchNormalization(Conv(x2, 96, { 3,3 }, { 1, 1 }, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 96, { 3,3 }, { 1, 1 }, "same")));
    x2 = BatchNormalization(Conv(UpSampling(x2, {2,2}), 96, {2,2}, {1,1}, "same"));

    //x1 = Concat({x1,x2});
    x1 = x2;
    x1 = ReLu(BatchNormalization(Conv(x1, 64, {3,3}, {1,1}, "same")));
    x1 = ReLu(BatchNormalization(Conv(x1, 64, {3,3}, {1,1}, "same")));
    y = Conv(x1, num_classes, {1,1});
    y = Sigmoid(y);

    return y;
}
layer UNetWithPaddingBN_v003(layer x, const int& num_classes)
{
    layer x1_3, x1_5, x1_7;
    layer x1, x1b;
    layer x2, x2a, x2b;
    layer x3, x3a, x3b;
    layer x4, x4a;
    layer x5;
    layer z1, z2, z3, z4, z5;
    layer y;

    x1_3 = ReLu(BatchNormalization(Conv(x, 32, {3,3}, {1,1}, "same")));
    x1_5 = ReLu(BatchNormalization(Conv(x, 32, {5,5}, {1,1}, "same")));
    x1_7 = ReLu(BatchNormalization(Conv(x, 32, {7,7}, {1,1}, "same")));

    x1 = Concat({x1_3, x1_5, x1_7});

    x1 = ReLu(BatchNormalization(Conv(x1, 64, {3,3}, {1,1}, "same")));

    x1b = MaxPool(x1, {2,2}, {2,2});
    x1b = ReLu(BatchNormalization(Conv(x1b, 64, {1,1}, {1,1}, "same")));

    x2 = MaxPool(x1, {2,2}, {2,2});
    x2 = ReLu(BatchNormalization(Conv(x2, 64, {3,3}, {1,1}, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 64, {3,3}, {1,1}, "same")));

    x2a = UpSampling(x2, {2,2});
    x2a = ReLu(BatchNormalization(Conv(x2a, 64, {1,1}, {1,1}, "same")));
    x2b = MaxPool(x2, {2,2}, {2,2});
    x2b = ReLu(BatchNormalization(Conv(x2b, 64, {1,1}, {1,1}, "same")));

    x3 = MaxPool(x2, {2,2}, {2,2});
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 128, {3,3}, {1,1}, "same")));

    x3a = UpSampling(x3, {2,2});
    x3a = ReLu(BatchNormalization(Conv(x3a, 128, {3,3}, {1,1}, "same")));
    x3b = MaxPool(x3, {2,2}, {2,2});
    x3b = ReLu(BatchNormalization(Conv(x3b, 128, {3,3}, {1,1}, "same")));

    x4 = MaxPool(x3, {2,2}, {2,2});
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 256, {3,3}, {1,1}, "same")));

    x4a = UpSampling(x4, {2,2});
    x4a = ReLu(BatchNormalization(Conv(x4a, 256, {3,3}, {1,1}, "same")));

    x5 = MaxPool(x4, {2,2}, {2,2});
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    x5 = ReLu(BatchNormalization(Conv(x5, 512, {3,3}, {1,1}, "same")));
    z5 = BatchNormalization(Conv(UpSampling(x5, {2,2}), 256, {2,2}, {1, 1}, "same"));

    ///////////////////////////////////////////////////////////////////////////////

    z4 = Concat({x3b, x4, z5});
    z4 = ReLu(BatchNormalization(Conv(z4, 256, {3,3}, {1,1}, "same")));
    z4 = ReLu(BatchNormalization(Conv(z4, 256, {3,3}, {1,1}, "same")));
    z4 = BatchNormalization(Conv(UpSampling(z4, {2,2}), 128, {2,2}, {1,1}, "same"));

    z3 = Concat({x4a, x2b, z4});
    z3 = ReLu(BatchNormalization(Conv(z3, 128, {3,3}, {1,1}, "same")));
    z3 = ReLu(BatchNormalization(Conv(z3, 128, {3,3}, {1,1}, "same")));
    z3 = BatchNormalization(Conv(UpSampling(z3, {2,2}), 96, {2,2}, {1, 1}, "same"));

    z2 = Concat({x3a, x1b, z3});
    z2 = ReLu(BatchNormalization(Conv(z2, 96, { 3,3 }, { 1, 1 }, "same")));
    z2 = ReLu(BatchNormalization(Conv(z2, 96, { 3,3 }, { 1, 1 }, "same")));
    z2 = BatchNormalization(Conv(UpSampling(z2, {2,2}), 96, {2,2}, {1,1}, "same"));

    z1 = Concat({x2a, z2});
    z1 = ReLu(BatchNormalization(Conv(z1, 64, {3,3}, {1,1}, "same")));
    z1 = ReLu(BatchNormalization(Conv(z1, 64, {3,3}, {1,1}, "same")));
    y = Conv(z1, num_classes, {1,1});
    y = Sigmoid(y);

    return y;
}

layer ResNet_01(layer x, const int& num_classes)
{
    layer l, l0, l1, l2, l3;

    l1 = ReLu(BatchNormalization(Conv(x, 32, { 3,3 }, { 1,1 }, "same")));
    l2 = ReLu(BatchNormalization(Conv(x, 32, { 5,5 }, { 1,1 }, "same")));
    l3 = ReLu(BatchNormalization(Conv(x, 32, { 7,7 }, { 1,1 }, "same")));

    l = Concat({ l1,l2,l3 });

    for (int filters : {64, 64, 128, 128, 256, 512}) {
        l = Conv(l, 1, { 3,3 }, { 1,1 }, "same");
        l0 = MaxPool(l, { 2,2 }, { 2,2 });

        l1 = ReLu(BatchNormalization(Conv(l0, filters, { 3,3 }, { 1,1 }, "same")));
        l2 = ReLu(BatchNormalization(Conv(l1, filters, { 3,3 }, { 1,1 }, "same")));
        l3 = ReLu(BatchNormalization(Conv(l2, filters, { 3,3 }, { 1,1 }, "same")));

        l = Concat({ l1,l2,l3 });
    }

    l = Reshape(l, { -1 });
    l = ReLu(BatchNormalization(Dense(l, 4096)));
    l = ReLu(BatchNormalization(Dense(l, 2048)));
    l = Softmax(Dense(l, num_classes));

    return l;
}
