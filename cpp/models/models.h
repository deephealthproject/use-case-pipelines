#ifndef MODELS_H_
#define MODELS_H_

#include "ecvl/support_eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

// Model LeNet (same as https://github.com/pytorch/examples/blob/master/mnist/main.py)
eddl::layer LeNet(eddl::layer x, const int& num_classes);

// Model VGG16
eddl::layer VGG16(eddl::layer x, const int& num_classes);
eddl::layer VGG16_inception_1(eddl::layer x, const int& num_classes);
eddl::layer VGG16_inception_2(eddl::layer x, const int& num_classes);

// Model SegNet (https://mi.eng.cam.ac.uk/projects/segnet)
eddl::layer SegNet(eddl::layer x, const int& num_classes);

// Model U-Net (https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net)
eddl::layer UNet(eddl::layer x, const int& num_classes);

eddl::layer ResNet50(eddl::layer x, const int& num_classes);
eddl::layer ResNet101(eddl::layer x, const int& num_classes);
eddl::layer ResNet152(eddl::layer x, const int& num_classes);

eddl::layer Nabla(eddl::layer x, const int& num_classes);

// From https://github.com/jfzhang95/pytorch-deeplab-xception
// Depth-wise separable convolutions have been removed
class DeepLabV3Plus
{
    int num_classes_;
    bool pretrained_ = true;
    const int block_expansion_ = 4;
    int inplanes_ = 64, output_layers_ = 256;
    float bn_momentum = 0.9f, bn_eps = 1e-5f;

    eddl::layer bottleneck(eddl::layer x, int planes, int stride = 1, bool downsample = false, int dilation = 1, bool conv = false, bool norm = false)
    {
        eddl::layer residual = x;
        eddl::layer out = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, planes, { 1, 1 }, { 1, 1 }, "same", false), true));
        out = eddl::Pad(out, std::vector<int>(4, dilation));
        out = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(out, planes, { 3,3 }, { stride,stride }, "valid", false, 1, { dilation,dilation }), true));
        out = eddl::BatchNormalization(eddl::Conv2D(out, planes * block_expansion_, { 1, 1 }, { 1, 1 }, "same", false), true);
        if (downsample) {
            residual = eddl::BatchNormalization(eddl::Conv2D(x, planes * block_expansion_, { 1, 1 }, { stride,stride }, "same", false), true);
        }
        out = eddl::Add({ out,residual });
        out = eddl::ReLu(out);
        return out;
    }

    eddl::layer make_layer(eddl::layer x, int planes, int blocks, int stride = 1, int dilation = 1)
    {
        bool downsample = false;
        if (stride != 1 || inplanes_ != planes * block_expansion_) {
            downsample = true;
        }

        x = bottleneck(x, planes, stride, downsample, dilation);
        inplanes_ = planes * block_expansion_;
        for (int i = 1; i < blocks; ++i) {
            x = bottleneck(x, planes, 1, false, dilation);
        }
        return x;
    }
    eddl::layer make_layer(eddl::layer x, int planes, const std::vector<int>& blocks, int stride = 1, int dilation = 1)
    {
        bool downsample = false;
        if (stride != 1 || inplanes_ != planes * block_expansion_) {
            downsample = true;
        }

        x = bottleneck(x, planes, stride, downsample, blocks[0] * dilation);
        inplanes_ = planes * block_expansion_;
        for (int i = 1; i < blocks.size(); ++i) {
            x = bottleneck(x, planes, 1, false, blocks[i] * dilation);
        }
        return x;
    }
    std::vector<eddl::layer> ResNet(eddl::layer x, const std::vector<int>& layers, int output_stride)
    {
        std::vector<int> strides;
        std::vector<int> dilations;
        if (output_stride == 16) {
            strides = { 1, 2, 2, 1 };
            dilations = { 1, 1, 1, 2 };
        }
        else if (output_stride == 8) {
            strides = { 1, 2, 1, 1 };
            dilations = { 1, 1, 2, 4 };
        }
        else {
            throw "Not implemented output_stride";
        }
        x = eddl::Pad(x, { 3,3,3,3 });
        x = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, 64, { 7,7 }, { 2,2 }, "none", false), true));
        x = eddl::Pad(x, { 1,1,1,1 });
        x = eddl::MaxPool2D(x, { 3,3 }, { 2,2 }, "none");
        x = make_layer(x, 64, layers[0], strides[0], dilations[0]);
        eddl::layer low_level_feat = x;
        x = make_layer(x, 128, layers[1], strides[1], dilations[1]);
        x = make_layer(x, 256, layers[2], strides[2], dilations[2]);
        x = make_layer(x, 512, { 1, 2, 4 }, strides[3], dilations[3]);

        return { x, low_level_feat };
    }

    std::vector<eddl::layer> ResNet101(eddl::layer x, int output_stride)
    {
        return ResNet(x, { 3,4,23,3 }, output_stride);
    }

    eddl::layer ASPPModule(eddl::layer x, int dilation)
    {
        //x = eddl::DepthwiseConv2D(x, { 3, 3 }, { 1,1 }, "same", false, { dilation,dilation });
        int filters = x->output->shape[1];  // one filter per channel (...with depth D)
        x = eddl::Conv2D(x, filters, { 3,3 }, { 1,1 }, "same", false, filters, { dilation, dilation }); // DepthwiseConv2D but with padding
        x = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, output_layers_, { 1,1 }, { 1,1 }, "same", false), true, bn_momentum, bn_eps));
        return x;
    }
    eddl::layer ASPP(eddl::layer x, int output_stride, int& scale_factor)
    {
        std::vector<int> dilations;
        if (output_stride == 16) {
            dilations = { 1, 6, 12, 18 };
            scale_factor = 4;
        }
        else if (output_stride == 8) {
            dilations = { 1, 12, 24, 36 };
            scale_factor = 2;
        }
        else {
            throw std::runtime_error(ECVL_ERROR_MSG "Not implemented output_stride");
        }

        eddl::layer x1 = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, output_layers_, { 1,1 }, { 1,1 }, "same", false, 1, { 1,1 }), true, bn_momentum, bn_eps));
        eddl::layer x2 = ASPPModule(x, dilations[1]);
        eddl::layer x3 = ASPPModule(x, dilations[2]);
        eddl::layer x4 = ASPPModule(x, dilations[3]);
        eddl::layer x5 = eddl::GlobalAveragePool2D(x);
        x5 = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x5, output_layers_, { 1,1 }, { 1,1 }, "same", false), true, bn_momentum, bn_eps));
        x5 = eddl::UpSampling2D(x5, { x4->getShape()[2], x4->getShape()[3] });
        x = eddl::Concat({ x1,x2,x3,x4,x5 });
        x = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, output_layers_, { 1,1 }, { 1,1 }, "same", false), true, bn_momentum, bn_eps));
        x = eddl::Dropout(x, 0.5f);
        x = ASPPModule(x, dilations[0]);

        return x;
    }

    eddl::layer Decoder(eddl::layer x, eddl::layer low_level_feat)
    {
        low_level_feat = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(low_level_feat, 48, { 1,1 }, { 1,1 }, "same", false), true, bn_momentum, bn_eps));
        x = eddl::Concat({ x, low_level_feat });

        int filters = x->output->shape[1];  // one filter per channel (...with depth D)
        x = eddl::Conv2D(x, filters, { 3,3 }, { 1,1 }, "same", false, filters, { 1, 1 }); // DepthwiseConv2D but with padding
        //x = eddl::DepthwiseConv2D(x, { 3, 3 }, { 1,1 }, "same", false, { 1,1 });
        x = eddl::ReLu(eddl::BatchNormalization(eddl::Conv2D(x, output_layers_, { 1,1 }, { 1,1 }, "same", false), true, bn_momentum, bn_eps));
        x = eddl::Conv2D(x, num_classes_, { 1,1 }, { 1,1 }, "same", false);
        return x;
    }

public:

    DeepLabV3Plus(int num_classes = 1, bool pretrained = true) : num_classes_{ num_classes }, pretrained_{ pretrained } {}

    eddl::layer forward(eddl::layer& input, int output_stride = 16)
    {
        eddl::layer x, low_level_feat;
        int scale_factor = 4; // if output_stride is 8 it will be modified to 2 in the ASPP module
        if (pretrained_) {
            auto resnet101 = import_net_from_onnx_file("resnet101.onnx", { input->getShape()[1], input->getShape()[2], input->getShape()[3] });
            input = eddl::getLayer(resnet101, "input"); // set input layer
            x = eddl::getLayer(resnet101, "Relu_341");
            low_level_feat = eddl::getLayer(resnet101, "Relu_35");
        }
        else {
            std::vector<eddl::layer> backbone = ResNet101(input, output_stride);
            x = backbone[0];
            low_level_feat = backbone[1];
        }

        x = ASPP(x, output_stride, scale_factor);
        x = eddl::UpSampling2D(x, { scale_factor,scale_factor });
        x = Decoder(x, low_level_feat);
        x = eddl::UpSampling2D(x, { 4,4 });
        x = eddl::Sigmoid(x);

        return x;
    }
};
#endif // MODELS_H_