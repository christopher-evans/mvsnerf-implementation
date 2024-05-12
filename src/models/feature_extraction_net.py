import torch.nn as nn
from inplace_abn import InPlaceABN as BatchNormActivation


class FeatureExtractionNet(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractionNet, self).__init__()

        self.convolutions = nn.Sequential(
            # initial convolutional layer
            nn.Conv2d(in_channels, 8, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(8),
            nn.Conv2d(8, 8, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(8),

            # first down-sample layer
            nn.Conv2d(8, 16, 5, stride=2, padding=2, bias=False),
            BatchNormActivation(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(16),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(16),

            # second down-sample layer
            nn.Conv2d(16, 32, 5, stride=2, padding=2, bias=False),
            BatchNormActivation(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(32),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            BatchNormActivation(32),

            # top layer
            nn.Conv2d(32, 32, 1)
        )

    def forward(self, x):
        batch_size, viewpoints, channels, height, width = x.shape

        features = self.convolutions(x.reshape(batch_size * viewpoints, channels, height, width))
        return features.view(batch_size, viewpoints, *features.shape[1:])
