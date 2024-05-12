import lightning as pl
from torch import optim
from dataclasses import dataclass
from models.smooth_l1_loss import SmoothL1Loss
from models.feature_extraction_net import FeatureExtractionNet
from models.positional_encoding import PositionalEncoding
from models.renderer import Renderer
from models.volume_encoding_net import VolumeEncodingNet
from utils.cost_volume import build_volume_features


@dataclass
class HyperParameters:
    learning_rate: float


class MVSNeRF(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.loss = SmoothL1Loss()
        # TODO parameter for out channels
        self.feature_extraction = FeatureExtractionNet()
        # TODO parameters for these
        self.volume_encoding = VolumeEncodingNet(in_channels=32+9)
        self.self_attention_renderer = Renderer()

        self.save_hyperparameters()

    def configure_optimizers(self):
        eta_min = 1e-7
        learning_rate = 5e-4
        num_epochs = 4

        # TODO: what about betas
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = 0

        # mvs_images shape (batch_size, viewpoints, channels, height, width)
        mvs_images = batch['mvs_images']
        print('mvs_images ', mvs_images.shape)
        print('depth_bounds ', batch['depth_bounds'].shape)
        print('image_warp_matrices ', batch['image_warp_matrices'].shape)

        # first viewpoint is the reference view, pass in the source views
        image_features = self.feature_extraction(mvs_images)
        print('image_features ', image_features.shape)

        # build the volume features at source views with homography warping
        volume_features = build_volume_features(
            image_features,
            # no need for the reference image
            mvs_images[:, 1:],
            # wlog assume reference matrix is identity
            batch['image_warp_matrices'][:, 1:],
            # use depth bounds for reference frustrum, which we map back to source views
            batch['depth_bounds'][:, 0].squeeze(1),
            padding=12,
            depth_resolution=128
        )

        print('volume_features ', volume_features.shape)
        volume_encoding = self.volume_encoding(volume_features)

        # un-normalize images

        #radiance_field = self.self_attention_renderer(volume_encoding)
        #loss = loss + self.loss()

        self.log_metrics()

    def log_metrics(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass
        #print(batch)
