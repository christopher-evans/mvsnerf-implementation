import lightning as L
from torch import optim, nn, device


class MVSNeRF(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(2 * 2, 4),
            nn.ReLU(),
            nn.Linear(4, 3)
        ).to(device("cuda"))

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     return optimizer

    def configure_optimizers(self):
        eta_min = 1e-7
        learning_rate = 5e-4
        num_epochs = 4
        optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        #print(batch['mvs_images'].shape)
        #print(batch['depth_maps'].shape)
        #print(batch['scan_id'])
        #print(batch['viewpoint_ids'])
        #print(batch['lighting_id'])
        # print(batch['mvs_images'].shape)
        # print(batch['depth_maps'].shape)
        # training_step defines the train loop.
        # it is independent of forward
        #print(batch)
        pass

    def validation_step(self, batch, batch_idx):
        pass
        #print(batch)
