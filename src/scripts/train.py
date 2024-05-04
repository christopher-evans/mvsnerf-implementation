"""
Define your training process here.
Reference: https://lightning.ai/docs/pytorch/stable/common/trainer.html
Example:
from ml.utils.constants import LOGGING_DIR

model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.fit(model, data_module=datamodule)
"""
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices

from models.mvsnerf import MVSNeRF
from datasets.dtu.datamodule import DTUDataModule

# init the autoencoder
mvs_nerf = MVSNeRF()

# setup data
dtu_datamodule = DTUDataModule()


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
torch.set_float32_matmul_precision('medium')

logger = TensorBoardLogger(".experiments", name="my_model")
trainer = L.Trainer(limit_train_batches=1, log_every_n_steps=2, max_epochs=1, accelerator="cuda", devices=find_usable_cuda_devices(1), logger=logger)
trainer.fit(mvs_nerf, datamodule=dtu_datamodule)