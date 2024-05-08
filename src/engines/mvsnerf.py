import lightning as pl
from torch import optim
from models.smooth_l1_loss import SmoothL1Loss
from models.feature_extraction_net import FeatureExtractionNet
from models.positional_encoding import PositionalEncoding
from models.renderer import Renderer
from models.volume_encoding_net import VolumeEncodingNet
from dataclasses import dataclass


@dataclass
class HyperParameters:
    learning_rate: float


class MVSNeRF(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.loss = SmoothL1Loss()
        self.feature_extraction = FeatureExtractionNet()
        self.volume_encoding = VolumeEncodingNet()
        self.self_attention_renderer = Renderer()

        self.save_hyperparameters()

    def configure_optimizers(self):
        eta_min = 1e-7
        learning_rate = 5e-4
        num_epochs = 4

        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = 0
        nb_views = self.hparams.nb_views
        H, W = batch['mvs_images'].shape[-2:]
        H, W = int(H), int(W)

        image_features = self.feature_extraction(batch)
        volume_features = self.volume_encoding(image_features)

        # un-normalize images

        predictions = self.self_attention_renderer(volume_features)
        loss = loss + self.loss()

        self.log_metrics()

    def log_metrics(self):
        pass

    def validation_step(self, batch, batch_idx):
        pass
        #print(batch)
