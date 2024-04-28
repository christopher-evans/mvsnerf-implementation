import numpy as np

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import DTUDataset, view_ids_top_three, view_ids_rand_top_five


def get_lighting_conditions(stage):
    """
    Fetch lighting conditions to use for images.  In the training process all lighting conditions 0-6 are
    used.  For testing lighting condition 6 is used, thought to be the brightest by the authors of MVSNeRF.

    :param stage:
    :return:
    """
    if stage == 'train':
        return list(range(7))

    return [3]


class DTUDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = ".data/processed/dtu_example",
        config_dir: str = ".configs/dtu_example/split_example",
        batch_size: int = 32,
        scale_factor=1.0 / 200,
        down_sample=1.0,
        max_length=-1
    ):
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
        self.viewpoint_ids = np.array([])

        # A list of all viewpoint IDs used by mvs_configurations
        # This is used to parse camera parameters
        self.viewpoint_index_map = np.array([])

        # camera projection matrices and mappings to and from world and camera co-ordinates
        self.projection_matrices, self.intrinsic_parameters = [], []
        self.world_to_cameras, self.cameras_to_worlds = [], []

    def setup(self, stage):
        """
        Given a model stage, set up data configuration from files.

        :param stage: 'fit', 'validate', 'test', or 'predict'
        """
        image_pairings = self.build_image_pairings()

        self.build_mvs_configurations(stage, image_pairings)
        self.build_camera_configurations()

    def build_image_pairings(self):
        """
        Build mapping from reference image to a list of source images, using image pairing configuration file.
        Populates a list of viewpoint IDs and an inverse map from viewpoint ID to array index.
        :return:
        """
        image_pairings = {}
        viewpoint_id_dict = {}

        # parse image pairing configuration file
        with open(f'{self.config_dir}/image_pairing.txt', encoding='utf-8') as image_pairing_config:
            # total number of viewpoints, should be 49
            num_viewpoint = int(image_pairing_config.readline())

            for _ in range(num_viewpoint):
                reference_view = int(image_pairing_config.readline().rstrip())
                source_views = [int(view_id) for view_id in image_pairing_config.readline().rstrip().split()[1::2]]

                image_pairings[reference_view] = source_views
                viewpoint_id_dict[reference_view] = True
                for source_view in source_views:
                    viewpoint_id_dict[source_view] = True

        # create viewpoint ID list
        self.viewpoint_ids = list(viewpoint_id_dict.keys())

        # create reverse map for viewpoint ID list
        self.viewpoint_index_map = np.zeros(np.max(self.viewpoint_ids)).astype(int)
        for index, viewpoint_id in enumerate(self.viewpoint_ids):
            self.viewpoint_index_map[viewpoint_id] = index

        return image_pairings

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

        lighting_conditions = get_lighting_conditions(stage)
        with open(f'{self.config_dir}/{stage}.txt', encoding='utf-8') as stage_scans_config:
            scans = [line.rstrip() for line in stage_scans_config.readlines()]

        for scan_id in scans:
            for reference_view in image_pairings:
                source_views = image_pairings[reference_view]

                for lighting_condition_id in lighting_conditions:
                    self.mvs_configurations += [(scan_id, lighting_condition_id, reference_view, source_views)]

    def parse_camera_config(self, config_file_name):
        with open(config_file_name, encoding='utf-8') as config_file:
            lines = [line.rstrip() for line in config_file.readlines()]

        # extrinsic parameters: lines 1-4 define a 4×4 matrix
        extrinsic_params = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsic_params = extrinsic_params.reshape((4, 4))

        # intrinsics parameters: lines 7-9 define a 3×3 matrix
        intrinsic_params = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsic_params = intrinsic_params.reshape((3, 3))

        # depth_min & depth_interval: line 11
        # TODO: why the factor of 192?
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor

        # TODO: define but not used? check
        # depth_interval = float(lines[11].split()[1])

        return intrinsic_params, extrinsic_params, [depth_min, depth_max]

    def build_camera_configurations(self):
        projection_matrices, intrinsic_parameters, world_to_cameras, cameras_to_worlds = [], [], [], []

        for viewpoint_id in self.viewpoint_ids:
            camera_config_file = f'{self.data_dir}/Cameras/train/{viewpoint_id:08d}_cam.txt'
            intrinsic_params, extrinsic_params, near_far = self.parse_camera_config(camera_config_file)

            intrinsic_params[:2] *= 4
            extrinsic_params[:3, 3] *= self.scale_factor

            intrinsic_params[:2] = intrinsic_params[:2] * self.down_sample
            intrinsic_parameters += [intrinsic_params.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic_params[:2] = intrinsic_params[:2] / 4
            proj_mat_l[:3, :4] = intrinsic_params @ extrinsic_params[:3, :4]

            projection_matrices += [(proj_mat_l, near_far)]
            world_to_cameras += [extrinsic_params]
            cameras_to_worlds += [np.linalg.inv(extrinsic_params)]

        self.projection_matrices, self.intrinsic_parameters = projection_matrices, intrinsic_parameters
        self.world_to_cameras, self.cameras_to_worlds = world_to_cameras, cameras_to_worlds

    def train_dataloader(self):
        return DataLoader(
            DTUDataset(
                self.mvs_configurations,
                view_ids_rand_top_five,
                self.viewpoint_index_map,
                self.projection_matrices,
                self.intrinsic_parameters,
                self.world_to_cameras,
                self.cameras_to_worlds,
                self.data_dir,
                self.max_length,
                self.scale_factor,
                self.down_sample,
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
                view_ids_top_three,
                self.viewpoint_index_map,
                self.projection_matrices,
                self.intrinsic_parameters,
                self.world_to_cameras,
                self.cameras_to_worlds,
                self.data_dir,
                self.max_length,
                self.scale_factor,
                self.down_sample,
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
                view_ids_top_three,
                self.viewpoint_index_map,
                self.projection_matrices,
                self.intrinsic_parameters,
                self.world_to_cameras,
                self.cameras_to_worlds,
                self.data_dir,
                self.max_length,
                self.scale_factor,
                self.down_sample,
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
                view_ids_top_three,
                self.viewpoint_index_map,
                self.projection_matrices,
                self.intrinsic_parameters,
                self.world_to_cameras,
                self.cameras_to_worlds,
                self.data_dir,
                self.max_length,
                self.scale_factor,
                self.down_sample,
            ),
            shuffle=False,
            num_workers=4,
            batch_size=self.batch_size,
            pin_memory=True
        )
