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

    return x;
}