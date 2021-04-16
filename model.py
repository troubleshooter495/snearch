import torch
import torch.nn as nn


class AE3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.encoder_seq = nn.Sequential()

        self.encoder_seq.add_module('conv1',
                                    nn.Conv2d(in_channels=3, out_channels=6,
                                              kernel_size=3,
                                              padding=1))
        self.encoder_seq.add_module('bn_conv1', nn.BatchNorm2d(6))
        self.encoder_seq.add_module('conv1_relu', nn.ReLU())
        self.encoder_seq.add_module('pool1',
                                    nn.MaxPool2d(kernel_size=3, stride=3))

        self.encoder_seq.add_module('conv1-5',
                                    nn.Conv2d(in_channels=6, out_channels=6,
                                              kernel_size=2, padding=1))
        self.encoder_seq.add_module('conv1-5_relu', nn.ReLU())

        self.encoder_seq.add_module('conv2',
                                    nn.Conv2d(in_channels=6, out_channels=10,
                                              kernel_size=3,
                                              padding=1))
        self.encoder_seq.add_module('conv2_relu', nn.ReLU())
        self.encoder_seq.add_module('pool2',
                                    nn.MaxPool2d(kernel_size=3, stride=3,
                                                 padding=0))

        self.encoder_seq.add_module('conv2-5',
                                    nn.Conv2d(in_channels=10, out_channels=10,
                                              kernel_size=2, padding=1))
        self.encoder_seq.add_module('conv2-5_relu', nn.ReLU())

        self.encoder_seq.add_module('conv3',
                                    nn.Conv2d(in_channels=10, out_channels=16,
                                              kernel_size=3,
                                              padding=1))
        self.encoder_seq.add_module('bn_conv2', nn.BatchNorm2d(16))
        self.encoder_seq.add_module('conv3_relu', nn.ReLU())
        self.encoder_seq.add_module('pool3',
                                    nn.MaxPool2d(kernel_size=2, stride=2,
                                                 padding=0))

        self.decoder_seq = nn.Sequential()

        self.decoder_seq.add_module('deconv1',
                                    nn.ConvTranspose2d(in_channels=16,
                                                       out_channels=10,
                                                       kernel_size=2, stride=2,
                                                       padding=0))
        self.decoder_seq.add_module('deconv1_relu', nn.ReLU())

        self.decoder_seq.add_module('deconv1-5',
                                    nn.ConvTranspose2d(in_channels=10,
                                                       out_channels=10,
                                                       kernel_size=2,
                                                       padding=0))
        self.decoder_seq.add_module('deconv1-5_relu', nn.ReLU())
        self.decoder_seq.add_module('bn_deconv1', nn.BatchNorm2d(10))

        self.decoder_seq.add_module('deconv2',
                                    nn.ConvTranspose2d(in_channels=10,
                                                       out_channels=6,
                                                       kernel_size=3, stride=3,
                                                       padding=1))  # convolution
        self.decoder_seq.add_module('deconv2_relu', nn.ReLU())

        self.decoder_seq.add_module('deconv2-5',
                                    nn.ConvTranspose2d(in_channels=6,
                                                       out_channels=6,
                                                       kernel_size=2,
                                                       padding=2))
        self.decoder_seq.add_module('deconv2-5_relu', nn.ReLU())
        self.decoder_seq.add_module('bn_deconv2', nn.BatchNorm2d(6))

        self.decoder_seq.add_module('deconv3', nn.ConvTranspose2d(in_channels=6,
                                                                  out_channels=3,
                                                                  kernel_size=3,
                                                                  stride=3,
                                                                  padding=(0,
                                                                           1)))

        self.alpha = nn.Parameter(torch.FloatTensor([1]))
        self.bias = nn.Parameter(torch.FloatTensor([0]))

    def encode(self, x):
        self.encoded = self.encoder_seq(x)
        return self.encoded

    def decode(self, x):
        self.decoded = self.decoder_seq(x) * self.alpha + self.bias
        return self.decoded

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


def load_model(path):
    return torch.load("../" + path, map_location=torch.device('cpu'))