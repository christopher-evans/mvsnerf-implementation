import torch.nn as nn
from inplace_abn import InPlaceABN as BatchNormActivation
import logging


def weights_init(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight)

        if module.bias is not None:
            nn.init.zeros_(module.bias)


class VolumeEncodingNet(nn.Module):
    def __init__(self, in_channels=3):
        super(VolumeEncodingNet, self).__init__()

        self.first_convolution = nn.Sequential(
            nn.Conv3d(in_channels, 8, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(8),
        )

        self.first_down_sample = nn.Sequential(
            nn.Conv3d(8, 16, 3, stride=2, padding=1, bias=False),
            BatchNormActivation(16),
            nn.Conv3d(16, 16, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(16),
        )
        self.second_down_sample = nn.Sequential(
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            BatchNormActivation(32),
            nn.Conv3d(32, 32, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(32),
        )
        self.third_down_sample = nn.Sequential(
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            BatchNormActivation(64),
            nn.Conv3d(64, 64, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(64),
        )

        self.first_up_sample = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
            BatchNormActivation(32),
        )
        self.second_up_sample = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1, bias=False),
            BatchNormActivation(16),
        )
        self.third_up_sample = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1, output_padding=1, bias=False),
            BatchNormActivation(8),
        )

    def init_weights(self):
        self.apply(weights_init)

    def forward(self, x):
        _, _, _, height, width = x.shape
        if height % 8 != 0 or width % 8 != 0:
            logging.getLogger(__name__) \
                .warning(f'Caution: width is {width} and height is {height}: should be divisible by 8')

        first_layer = self.first_convolution(x)
        first_down_sample = self.first_down_sample(first_layer)
        second_down_sample = self.second_down_sample(first_down_sample)

        x = self.third_down_sample(second_down_sample)

        x = second_down_sample + self.first_up_sample(x)
        del second_down_sample

        x = first_down_sample + self.second_up_sample(x)
        del first_down_sample

        x = first_layer + self.third_up_sample(x)
        del first_layer

        return x
