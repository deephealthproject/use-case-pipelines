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

layer BottleNeck(layer x, int planes, int stride, bool downsample = false)
{
    layer identity = x;
    // in_layer, out_channels, kernel, stride, padding, bias, groups, dilation
    x = BatchNormalization(Conv(x, planes, { 1,1 }, { 1,1 }, "same", false));
    x = BatchNormalization(Conv(x, planes, { 3,3 }, { stride,stride }, "same", false));
    x = ReLu(BatchNormalization(Conv(x, planes * 4, { 1,1 }, { 1,1 }, "same", false)));
    if (downsample) {
        identity = BatchNormalization(Conv(identity, planes * 4, { 1,1 }, { stride,stride }, "same", false));
    }
    return ReLu(Add({ identity, x }));
}

layer MakeResnet(layer x, const vector<pair<int, int>>& filters)
{
    x = ReLu(BatchNormalization(Conv(x, 64, { 7, 7 }, { 2, 2 }, "same", false)));
    x = MaxPool(x, { 3, 3 }, { 2, 2 }, "same");
    int i = 0;
    for (const auto& blocks : filters) {
        x = BottleNeck(x, blocks.second, 2 - (i == 0), true);
        for (i = 1; i < blocks.first; ++i) {
            x = BottleNeck(x, blocks.second, 1);
        }
    }
    x = AveragePool(x, { 7,7 }, { 1, 1 });
    x = Reshape(x, { -1 });
    return x;
}

layer ResNet50(layer x, const int& num_classes)
{
    vector<pair<int, int>> filters{ { 3, 64 },
                                  { 4, 128 },
                                  { 6, 256 },
                                  { 3, 512 } };
    x = MakeResnet(x, filters);
    x = Softmax(Dense(x, num_classes));
    return x;
}

layer ResNet101(layer x, const int& num_classes)
{
    vector<pair<int, int>> filters{ { 3, 64 },
                                  { 4, 128 },
                                  { 23, 256 },
                                  { 3, 512 } };
    x = MakeResnet(x, filters);
    x = Softmax(Dense(x, num_classes));
    return x;
}

layer ResNet152(layer x, const int& num_classes)
{
    vector<pair<int, int>> filters{ { 3, 64 },
                                  { 8, 128 },
                                  { 36, 256 },
                                  { 3, 512 } };
    x = MakeResnet(x, filters);
    x = Softmax(Dense(x, num_classes));
    return x;
}