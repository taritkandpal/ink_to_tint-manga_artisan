"""
Class definitions for GAN Discriminator.
"""
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """
    Convolution block for Discriminator.
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            # bias is False since batch normalization takes care of the "shift" variable
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    Discriminator for GAN.
    """

    def __init__(self, in_channels=4, features=[32, 64, 96, 128, 192, 256, 384, 512]):
        super(Discriminator, self).__init__()

        # list to append convolution blocks
        layers = []

        # initial layer is defined without batch normalization (since the input image is already normalized)
        initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        layers.append(initial)

        # intermediate layers
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_channels, feature, stride=2))
            in_channels = feature

        # last output layer
        layers.append(CNNBlock(in_channels, 1, stride=1))

        # unfold all the layers defined
        self.model = nn.Sequential(*layers)

    def forward(self, bw_image, color_image):
        """
        Discriminator forward pass.
        """
        combined = torch.cat([bw_image, color_image], dim=1)
        combined = self.model(combined)
        return combined


if __name__ == "__main__":
    dis = Discriminator()
    op = dis(torch.randn(2, 1, 512, 512), torch.randn(2, 3, 512, 512))
    print(op.shape)
