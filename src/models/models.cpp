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

layer UNet(layer x, const int& num_classes)
{
    layer x2;
    layer x3;
    layer x4;
    layer x5;

    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "none"), "relu");
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "none"), "relu");
    x2 = MaxPool(x, { 2,2 }, { 2,2 });
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "none"), "relu");
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "none"), "relu");
    x3 = MaxPool(x2, { 2,2 }, { 2,2 });
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "none"), "relu");
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "none"), "relu");
    x4 = MaxPool(x3, { 2,2 }, { 2,2 });
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "none"), "relu");
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "none"), "relu");
    x5 = MaxPool(x4, { 2,2 }, { 2,2 });
    x5 = Activation(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "none"), "relu");
    x5 = Activation(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "none"), "relu");
    x5 = Conv(UpSampling(x5, { 2,2 }), 512, { 1,1 }, { 1, 1 }, "none"); // Should be 2x2 Conv
    
    x4 = Crop(x4, { 5, 5 }, { 60, 60 }, true);
    x4 = Concat({ x4, x5 });
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "none"), "relu");
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "none"), "relu");
    x4 = Conv(UpSampling(x4, { 2,2 }), 256, { 1,1 }, { 1, 1 }, "none"); // Should be 2x2 Conv

    x3 = Crop(x3, { 17, 17 }, { 120, 120 }, true);
    x3 = Concat({ x3, x4 });
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "none"), "relu");
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "none"), "relu");
    x3 = Conv(UpSampling(x3, { 2,2 }), 128, { 1,1 }, { 1, 1 }, "none"); // Should be 2x2 Conv

    x2 = Crop(x2, { 41, 41 }, { 240, 240 }, true);
    x2 = Concat({ x2, x3 });
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "none"), "relu");
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "none"), "relu");
    x2 = Conv(UpSampling(x2, { 2,2 }), 64, { 1,1 }, { 1, 1 }, "none"); // Should be 2x2 Conv

    x = Crop(x, { 89, 89 }, { 480, 480 }, true);
    x = Concat({ x, x2 });
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "none"), "relu");
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "none"), "relu");
    x = Conv(x, 3, { 1,1 }); // Should be num_classes instead of 3
    
    return x;
}

layer UNetWithPadding(layer x, const int& num_classes)
{
    layer x2;
    layer x3;
    layer x4;
    layer x5;

    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"), "relu");
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"), "relu");
    x2 = MaxPool(x, { 2,2 }, { 2,2 });
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"), "relu");
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"), "relu");
    x3 = MaxPool(x2, { 2,2 }, { 2,2 });
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"), "relu");
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"), "relu");
    x4 = MaxPool(x3, { 2,2 }, { 2,2 });
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"), "relu");
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"), "relu");
    x5 = MaxPool(x4, { 2,2 }, { 2,2 });
    x5 = Activation(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same"), "relu");
    x5 = Activation(Conv(x5, 1024, { 3,3 }, { 1, 1 }, "same"), "relu");
    x5 = Conv(UpSampling(x5, { 2,2 }), 512, { 3,3 }, { 1, 1 }, "same"); // Should be 2x2 Conv

    //x4 = Crop(x4, { 5, 5 }, { 60, 60 }, true);
    x4 = Concat({ x4, x5 });
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"), "relu");
    x4 = Activation(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same"), "relu");
    x4 = Conv(UpSampling(x4, { 2,2 }), 256, { 3,3 }, { 1, 1 }, "same"); // Should be 2x2 Conv

    //x3 = Crop(x3, { 17, 17 }, { 120, 120 }, true);
    x3 = Concat({ x3, x4 });
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"), "relu");
    x3 = Activation(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same"), "relu");
    x3 = Conv(UpSampling(x3, { 2,2 }), 128, { 3,3 }, { 1, 1 }, "same"); // Should be 2x2 Conv

    //x2 = Crop(x2, { 41, 41 }, { 240, 240 }, true);
    x2 = Concat({ x2, x3 });
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"), "relu");
    x2 = Activation(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same"), "relu");
    x2 = Conv(UpSampling(x2, { 2,2 }), 64, { 3,3 }, { 1, 1 }, "same"); // Should be 2x2 Conv

    //x = Crop(x, { 89, 89 }, { 480, 480 }, true);
    x = Concat({ x, x2 });
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"), "relu");
    x = Activation(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"), "relu");
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
    x5 = BatchNormalization(Conv(UpSampling(x5, { 2,2 }), 512, { 3,3 }, { 1, 1 }, "same")); // Should be 2x2 Conv

    //x4 = Crop(x4, { 5, 5 }, { 60, 60 }, true);
    x4 = Concat({ x4, x5 });
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x4 = ReLu(BatchNormalization(Conv(x4, 512, { 3,3 }, { 1, 1 }, "same")));
    x4 = BatchNormalization(Conv(UpSampling(x4, { 2,2 }), 256, { 3,3 }, { 1, 1 }, "same")); // Should be 2x2 Conv

    //x3 = Crop(x3, { 17, 17 }, { 120, 120 }, true);
    x3 = Concat({ x3, x4 });
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x3 = ReLu(BatchNormalization(Conv(x3, 256, { 3,3 }, { 1, 1 }, "same")));
    x3 = BatchNormalization(Conv(UpSampling(x3, { 2,2 }), 128, { 3,3 }, { 1, 1 }, "same")); // Should be 2x2 Conv

    //x2 = Crop(x2, { 41, 41 }, { 240, 240 }, true);
    x2 = Concat({ x2, x3 });
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x2 = ReLu(BatchNormalization(Conv(x2, 128, { 3,3 }, { 1, 1 }, "same")));
    x2 = BatchNormalization(Conv(UpSampling(x2, { 2,2 }), 64, { 3,3 }, { 1, 1 }, "same")); // Should be 2x2 Conv

    //x = Crop(x, { 89, 89 }, { 480, 480 }, true);
    x = Concat({ x, x2 });
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = ReLu(BatchNormalization(Conv(x, 64, { 3,3 }, { 1, 1 }, "same")));
    x = Conv(x, num_classes, { 1,1 });

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

layer LeNet(layer x, const int& num_classes)
{
    x = MaxPool(Activation(Conv(x, 20, { 5,5 }), "relu"), { 2,2 }, { 2,2 });
    x = MaxPool(Activation(Conv(x, 50, { 5,5 }), "relu"), { 2,2 }, { 2,2 });
    x = Reshape(x, { -1 });
    x = Activation(Dense(x, 500), "relu");
    x = Activation(Dense(x, num_classes), "softmax");

    return x;
}

layer FakeNet(layer x, const int& num_classes)
{
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = UpSampling(x, { 2,2 });
    x = Conv(x, num_classes, { 3,3 }, { 1, 1 }, "same"); // Should be 2x2 Conv

    return x;
}