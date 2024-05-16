import lightning as pl
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.loggers import TensorBoardLogger

from engines.mvsnerf import MVSNeRF, HyperParameters
from datasets.dtu.datamodule import DTUDataModule, DTUParameters
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, RichProgressBar, ModelCheckpoint


def train(args):
    """
    Train MVSNeRF.

    :param args: Command line parameters
    :type args: argparse.Namespace
    """
    # init the autoencoder
    hparams = HyperParameters(
        initial_learning_rate=args.initial_learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        epoch_count=args.epoch_count,
        image_feature_padding=args.image_feature_padding,
        depth_resolution=args.depth_resolution,
        ray_direction_random_sampling=args.ray_direction_random_sampling,
        ray_march_count=args.ray_march_count,
        ray_march_sample_count=args.ray_march_sample_count
    )
    mvs_nerf = MVSNeRF(hparams)

    # setup data
    dataset_parameters = DTUParameters(
        data_dir=args.data_dir,
        data_config_dir=args.data_config_dir,
        batch_size=args.batch_size,
        data_max_length=args.data_max_length,
        depth_scale_factor=args.depth_scale_factor,
        image_down_sample=args.image_down_sample
    )
    dtu_datamodule = DTUDataModule(dataset_parameters)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    torch.set_float32_matmul_precision('medium')
    device_stats = DeviceStatsMonitor(cpu_stats=True)
    lr_monitor = LearningRateMonitor()
    progress_bar = RichProgressBar()

    logger = TensorBoardLogger(".experiments", name="mvsnerf")
    trainer = pl.Trainer(
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.epoch_count,
        accelerator="cuda",
        devices=find_usable_cuda_devices(1),
        callbacks=[device_stats, lr_monitor, progress_bar],
        logger=logger,
        enable_checkpointing=True,
        benchmark=True,
        precision=16
    )
    trainer.fit(mvs_nerf, datamodule=dtu_datamodule)

