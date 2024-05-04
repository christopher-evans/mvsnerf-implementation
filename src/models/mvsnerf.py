import lightning as L
from torch import optim, nn

class MVSNeRF(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        print(batch['scan_id'])
        print(batch['viewpoint_ids'])
        print(batch['lighting_id'])
        # print(batch['mvs_images'].shape)
        # print(batch['depth_maps'].shape)
        print()
        # training_step defines the train loop.
        # it is independent of forward
        #print(batch)

    def validation_step(self, batch, batch_idx):
        pass
        #print(batch)