import lightning as L
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices

from engines.mvsnerf import MVSNeRF, HyperParameters
from datasets.dtu.datamodule import DTUDataModule
from lightning.pytorch.callbacks import DeviceStatsMonitor

def train(args):
    """
    Train MVSNeRF.

    :param args: Command line parameters
    :type args: argparse.Namespace
    """
    # init the autoencoder
    hparams = HyperParameters(
        learning_rate=args.learning_rate
    )
    mvs_nerf = MVSNeRF(hparams)

    # setup data
    dtu_datamodule = DTUDataModule()

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    torch.set_float32_matmul_precision('medium')
    device_stats = DeviceStatsMonitor(cpu_stats=True)

    #logger = TensorBoardLogger(".experiments", name="my_model")
    trainer = L.Trainer(
        limit_train_batches=4,
        max_epochs=5,
        accelerator="cuda",
        devices=find_usable_cuda_devices(1),
        callbacks=[device_stats]
    )
    trainer.fit(mvs_nerf, datamodule=dtu_datamodule)

