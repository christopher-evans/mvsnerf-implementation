import torch.nn as nn


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

        self.loss = nn.SmoothL1Loss()

    def forward(self, depth_prediction, depth_ground_truth, mask=None):
        if mask is None:
            mask = depth_ground_truth > 0

        return self.loss(depth_prediction[mask], depth_ground_truth[mask]) * 2 ** (1 - 2)
