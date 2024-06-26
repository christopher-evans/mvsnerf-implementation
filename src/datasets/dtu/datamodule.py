from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from datasets.dtu.dataset import DTUDataset
from datasets.dtu.utils.image_pairings import load_image_pairings, SourceViews
from datasets.dtu.utils.camera_config import load_camera_matrices
from datasets.dtu.utils.scan_config import load_scans
from dataclasses import dataclass


@dataclass
class DTUParameters:
    data_dir: str
    data_config_dir: str
    batch_size: int
    data_max_length: int
    depth_scale_factor: float
    image_down_sample: float


@dataclass
class MvsConfiguration:
    """Configuration for an MVS problem."""
    scan_id: str
    lighting_condition_id: int
    target_view: int
    source_views: list


def get_lighting_conditions(stage):
    """
    Fetch lighting conditions to use for images.  In the training process all lighting conditions 0-6 are
    used.  For testing lighting condition 3 is used, thought to be the brightest by the authors of MVSNeRF.

    :param stage:
    :return:
    """
    if stage == 'fit':
        return list(range(7))

    return [3]


class DTUDataModule(LightningDataModule):
    def __init__(self, dtu_parameters):
        super().__init__()
        self.data_dir = dtu_parameters.data_dir.rstrip('/')
        self.config_dir = dtu_parameters.data_config_dir.rstrip('/')
        self.batch_size = int(dtu_parameters.batch_size)
        self.depth_scale_factor = float(dtu_parameters.depth_scale_factor)
        self.image_down_sample = float(dtu_parameters.image_down_sample)
        self.max_length = int(dtu_parameters.data_max_length)

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
        source_view_sampler = SourceViews.top_k()
        if stage != 'fit':
            source_view_sampler = SourceViews.top_k()
        image_pairings, all_viewpoint_ids = load_image_pairings(
            f'{self.config_dir}/image_pairing.txt',
            source_view_sampler
        )

        self.build_mvs_configurations(stage, image_pairings)
        self.build_camera_matrices(all_viewpoint_ids)

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
        scan_ids = load_scans(self.config_dir, stage)

        for scan_id in scan_ids:
            for target_view in image_pairings:
                source_views = image_pairings[target_view]

                for lighting_condition_id in lighting_conditions:
                    self.mvs_configurations += [MvsConfiguration(
                        scan_id=scan_id,
                        lighting_condition_id=lighting_condition_id,
                        target_view=target_view,
                        source_views=source_views
                    )]

    def build_camera_matrices(self, all_viewpoint_ids):
        for viewpoint_id in all_viewpoint_ids:
            camera_matrices = load_camera_matrices(
                self.data_dir,
                viewpoint_id,
                depth_scale_factor=self.depth_scale_factor,
                image_down_sample=self.image_down_sample
            )

            self.camera_matrices[viewpoint_id] = camera_matrices

    def train_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                depth_scale_factor=self.depth_scale_factor,
                image_down_sample=self.image_down_sample,
                max_length=self.max_length,
            ),
            shuffle=True,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                depth_scale_factor=self.depth_scale_factor,
                image_down_sample=self.image_down_sample,
                max_length=10,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                self.camera_matrices,
                data_dir=self.data_dir,
                depth_scale_factor=self.depth_scale_factor,
                image_down_sample=self.image_down_sample,
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
                depth_scale_factor=self.depth_scale_factor,
                image_down_sample=self.image_down_sample,
                max_length=self.max_length,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )
