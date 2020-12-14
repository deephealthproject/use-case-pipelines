from collections import OrderedDict

import torch
import torch.nn as nn


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class Nabla(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(Nabla, self).__init__()

        features = 64
        self.conv0 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(features*2, features, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(num_features=features)
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.bn2 = nn.BatchNorm2d(num_features=features)
        self.bn3 = nn.BatchNorm2d(num_features=features)
        self.bn4 = nn.BatchNorm2d(num_features=features)
        self.bn5 = nn.BatchNorm2d(num_features=features)
        self.bn6 = nn.BatchNorm2d(num_features=features)
        self.bn7 = nn.BatchNorm2d(num_features=features)
        self.bn8 = nn.BatchNorm2d(num_features=features)
        self.bn9 = nn.BatchNorm2d(num_features=features)
        self.bn10 = nn.BatchNorm2d(num_features=features)
        self.bn11 = nn.BatchNorm2d(num_features=features)
        self.bn12 = nn.BatchNorm2d(num_features=features)
        self.bn13 = nn.BatchNorm2d(num_features=features)
        self.bn14 = nn.BatchNorm2d(num_features=features)
        self.bn15 = nn.BatchNorm2d(num_features=features)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.maxpool5 = nn.MaxPool2d(2, 2)
        self.upsampling = nn.Upsample(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool1(x1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.maxpool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)

        # decoder
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.upsampling(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.upsampling(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x2 = self.upsampling(x)

        x = torch.cat((x1, x2), dim=1)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu(x)
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu(x)
        x = self.conv_final(x)

        return x
