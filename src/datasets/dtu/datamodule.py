from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from datasets.dtu.dataset import DTUDataset
from datasets.dtu.utils.image_pairings import load_image_pairings, SourceViews
from datasets.dtu.utils.camera_config import load_camera_matrices
from datasets.dtu.utils.scan_config import load_scans
from dataclasses import dataclass

@dataclass
class MvsConfiguration:
    """Configuration for an MVS problem."""
    scan_id: str
    lighting_condition_id: int
    reference_view: int
    source_views: list


def get_lighting_conditions(stage):
    """
    Fetch lighting conditions to use for images.  In the training process all lighting conditions 0-6 are
    used.  For testing lighting condition 6 is used, thought to be the brightest by the authors of MVSNeRF.

    :param stage:
    :return:
    """
    if stage == 'fit':
        return list(range(7))

    return [3]


class DTUDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".data/processed/dtu_example",
        config_dir: str = ".configs/dtu_example/split_example",
        batch_size: int = 2,
        scale_factor=1.0 / 200,
        down_sample=1.0,
        max_length=-1
    ):
        """
        :param data_dir:
        :param config_dir:
        :param batch_size:
        :param scale_factor:
        :param down_sample:
        :param max_length:
        """
        super().__init__()
        self.data_dir = data_dir.rstrip("/")
        self.config_dir = config_dir.rstrip("/")
        self.batch_size = int(batch_size)
        self.scale_factor = float(scale_factor)
        self.down_sample = float(down_sample)
        self.max_length = int(max_length)

        # MVS problem configurations
        self.mvs_configurations = []

        # A list of all viewpoint IDs used by mvs_configurations
        # This is used to parse camera parameters
        self.camera_matrices = dict()

    def setup(self, stage):
        """
        Given a model stage, set up data configuration from files.

        :param stage: 'fit', 'validate', 'test', or 'predict'
        """
        source_view_sampler = SourceViews.rand_k_top_n()
        if stage != 'fit':
            source_view_sampler = SourceViews.top_k()
        image_pairings = load_image_pairings(f'{self.config_dir}/image_pairing.txt', source_view_sampler)

        self.build_mvs_configurations(stage, image_pairings)
        self.build_camera_matrices(image_pairings)

    def build_mvs_configurations(self, stage, image_pairings):
        """
        An MVS configuration consists of a scan_id, a lighting condition, a reference view and a list of source views.
        A configuration is created for each scan, each viewpoint and each lighting condition.  This configuration will
        contain the top-10 scoring source views, from which N <= 10 can be drawn from randomly during training.

        :param stage:
        :param image_pairings:
        :return:
        """
        self.mvs_configurations = []

        # TODO: put lighting conditions somewhere
        lighting_conditions = get_lighting_conditions(stage)
        scan_config_file = f'{self.config_dir}/{stage.lower()}.txt'
        scan_ids = load_scans(scan_config_file, stage)

        for scan_id in scan_ids:
            for reference_view in image_pairings.iterator():
                source_views = image_pairings.fetch(reference_view)

                for lighting_condition_id in lighting_conditions:
                    self.mvs_configurations += [MvsConfiguration(
                        scan_id=scan_id,
                        lighting_condition_id=lighting_condition_id,
                        reference_view=reference_view,
                        source_views=source_views
                    )]

    def build_camera_matrices(self, image_pairings):
        viewpoint_ids = image_pairings.fetch_all_viewpoint_ids()

        for viewpoint_id in viewpoint_ids:
            camera_config_file = f'{self.data_dir}/Cameras/train/{viewpoint_id:08d}_cam.txt'
            camera_config = load_camera_matrices(
                camera_config_file,
                viewpoint_id,
                scale_factor=self.scale_factor,
                down_sample=self.down_sample
            )
            camera_matrices = camera_config.fetch_matrices()

            self.camera_matrices[viewpoint_id] = camera_matrices

    def train_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                scale_factor=self.scale_factor,
                down_sample=self.down_sample,
                max_length=self.max_length,
            ),
            shuffle=True,
            num_workers=8,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                scale_factor=self.scale_factor,
                down_sample=self.down_sample,
                max_length=self.max_length,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                scale_factor=self.scale_factor,
                down_sample=self.down_sample,
                max_length=self.max_length,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                scale_factor=self.scale_factor,
                down_sample=self.down_sample,
                max_length=self.max_length,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )
