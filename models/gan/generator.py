"""
Class definitions for GAN Generator.
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    """
    Residual block for Generator Upsampling and Downsampling.
    """

    def __init__(self, in_channels, out_channels, down=True):
        super(Block, self).__init__()
        if down:
            # downsampling block making use of Conv2d
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            self.res = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            # upsampling block making use of ConvTranspose2d
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            self.res = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.ConvTranspose2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

    def forward(self, x):
        # forward pass with skip connection
        x = self.conv(x)
        x = x + self.res(x)
        return x


# Residual Block
class ResidualBlock(nn.Module):
    """
    Residual block for generator bottleneck.
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        # forward pass with skip connection
        return x + self.block(x)


# Create Generator
class Generator(nn.Module):
    """
    Generator for GAN.
    """

    def __init__(self, in_channels=1, out_channels=3, features=64):
        super(Generator, self).__init__()
        # initial block shouldn't have batch normalization
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

        # Encoder - Downsampling
        self.down1 = Block(features, features * 2, down=True)  # 64 -> 128
        self.down2 = Block(features * 2, features * 4, down=True)  # 128 -> 256
        self.down3 = Block(features * 4, features * 8, down=True)  # 256 -> 512
        self.down4 = Block(features * 8, features * 8, down=True)  # 512 -> 512

        # Residual Bottleneck
        self.bottleneck = nn.Sequential(ResidualBlock(features * 8), nn.ReLU())

        # Decoder - Upsampling
        self.up1 = Block(features * 8, features * 8, down=False)
        self.up2 = Block(features * 8 * 2, features * 4, down=False)
        self.up3 = Block(features * 4 * 2, features * 2, down=False)
        self.up4 = Block(features * 2 * 2, features, down=False)

        # final output block
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        Generator forward pass.
        """
        # downsampling
        d1 = self.initial(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)

        # bottleneck
        bottleneck = self.bottleneck(d5)

        # upsampling
        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        final = self.final(torch.cat([u4, d1], dim=1))

        return final


if __name__ == "__main__":
    gen = Generator()
    op = gen(torch.randn(2, 1, 512, 512))
    print(op.shape)
