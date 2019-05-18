import torch
from transformer_net import ConvLayer, UpsampleConvLayer, ResidualBlock


class SmallTransformerNet(torch.nn.Module):
    def __init__(self):
        super(SmallTransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        # Residual layers
        self.res6 = ResidualBlock(64)
        self.res7 = ResidualBlock(64)
        # Upsampling Layers
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.res6(y)
        y = self.res7(y)
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


