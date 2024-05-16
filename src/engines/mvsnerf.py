import random

import lightning as pl
import cv2, torch
from torch import optim
from dataclasses import dataclass
from torch.nn import MSELoss
from torchmetrics.image import PeakSignalNoiseRatio
from models.feature_extraction_net import FeatureExtractionNet
from models.renderer import Renderer
from models.volume_encoding_net import VolumeEncodingNet
from utils.cost_volume import build_volume_features
from utils.ray_marching import create_ray_offsets_sampled, create_ray_offsets_row, march_rays
from utils.rendering import create_direction_vectors, interpolate_volume_encoding, interpolate_pixel_colours, parse_nerf
import random


@dataclass
class HyperParameters:
    initial_learning_rate: float
    minimum_learning_rate: float
    epoch_count: int
    image_feature_padding: int
    depth_resolution: int
    ray_direction_random_sampling: bool
    ray_march_count: int
    ray_march_sample_count: int


class MVSNeRF(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.loss = MSELoss()
        self.peak_signal_to_noise = PeakSignalNoiseRatio()
        # TODO parameter for out channels
        self.feature_extraction = FeatureExtractionNet()
        # TODO parameters for these
        self.volume_encoding = VolumeEncodingNet(in_channels=32+9)
        self.self_attention_renderer = Renderer()
        self.self_attention_renderer.init_weights()

        self.save_hyperparameters()

    def on_train_start(self):
        self.log_dict(vars(self.hparams))

    def configure_optimizers(self):
        minimum_learning_rate = self.hparams.minimum_learning_rate
        initial_learning_rate = self.hparams.initial_learning_rate
        epoch_count = self.hparams.epoch_count

        # TODO: what about betas
        optimizer = optim.Adam(self.parameters(), lr=initial_learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count, eta_min=minimum_learning_rate)
        return [optimizer], [scheduler]

    def un_preprocess(self, mvs_images, shape=(1, 1, 3, 1, 1)):
        # TODO combine this with the pre-processing, can be static
        # un-normalize MVS images for visualization
        device = mvs_images.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], device=device).view(*shape)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225], device=device).view(*shape)

        return (mvs_images - mean) / std

    def training_step(self, batch, batch_idx):
        # mvs_images.shape: (batch_size, viewpoints, channels, height, width)
        mvs_images = batch['mvs_images']

        # image_features.shape: (batch_size, viewpoints, feature_channels, height / 4, width / 4)
        image_features = self.feature_extraction(mvs_images)

        # build the volume features at source views with homography warping
        # volume_features.shape: (batch_size, feature_channels + source_viewpoints * image_channels, depth_resolution, height / 4, width / 4)

        volume_features = build_volume_features(
            image_features,
            # no need for the reference image
            mvs_images[:, 1:],
            # wlog assume reference matrix is identity
            batch['image_warp_matrices'][:, 1:],
            # use depth bounds for reference frustrum, which we map back to source views
            batch['depth_bounds'][:, 0].squeeze(1),
            padding=self.hparams.image_feature_padding,
            depth_resolution=self.hparams.depth_resolution
        )
        volume_encoding = self.volume_encoding(volume_features)

        # un-normalize images and create ray samples
        mvs_images = self.un_preprocess(mvs_images)
        batch_size, _, _, height, width = mvs_images.shape
        ray_count, ray_sample_count = self.hparams.ray_march_count, self.hparams.ray_march_sample_count
        ray_offset_function = create_ray_offsets_sampled(
            height,
            width,
            ray_count,
            batch_size=batch_size,
            dtype=mvs_images.dtype,
            device=mvs_images.device
        )

        all_ray_directions, all_ray_origins, all_depth_samples, all_point_samples, all_point_samples_ndc, all_image_colours = march_rays(
            mvs_images,
            batch['intrinsic_param_matrices'],
            batch['camera_to_world_matrices'],
            batch['world_to_camera_matrices'],
            ray_offset_function,
            ray_count,
            ray_sample_count,
            batch['depth_bounds'],
            padding=self.hparams.image_feature_padding
        )

        volume_directions = create_direction_vectors(
            all_ray_directions[:, 0],
            batch['world_to_camera_matrices'][:, 1],
            ray_sample_count
        )

        # volume_features.shape: (batch_size, ray_count, ray_sample_count, 8)
        volume_features = interpolate_volume_encoding(
            volume_encoding,
            all_point_samples_ndc
        )
        # source_image_colours.shape: (batch_size, channels * source_viewpoints, ray_count, ray_sample_count)
        source_image_colours = interpolate_pixel_colours(
            all_point_samples[:, 0],
            mvs_images[:, 1:],
            batch['world_to_camera_matrices'][:, 1:],
            batch['intrinsic_param_matrices'][:, 1:],
            batch['depth_bounds'][:, 1:],
        )

        prediction_colours, prediction_density = self.self_attention_renderer(
            all_point_samples_ndc.view(batch_size * ray_count * ray_sample_count, 3),
            volume_directions,
            volume_features,
            source_image_colours
        )

        # prediction_rgb.shape: (batch_size, ray_count, channels)
        prediction_rgb = parse_nerf(
            prediction_colours.view(batch_size, ray_count, ray_sample_count, 3),
            prediction_density.view(batch_size, ray_count, ray_sample_count)
        )

        # compute loss
        mean_squared_error = self.loss(prediction_rgb, all_image_colours[:, 0])
        self.log('train/loss', mean_squared_error, prog_bar=True)

        return mean_squared_error

    def log_metrics(self):
        pass

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            # mvs_images.shape: (batch_size, viewpoints, channels, height, width)
            mvs_images = batch['mvs_images']

            # image_features.shape: (batch_size, viewpoints, feature_channels, height / 4, width / 4)
            image_features = self.feature_extraction(mvs_images)

            # build the volume features at source views with homography warping
            # volume_features.shape: (batch_size, feature_channels + source_viewpoints * image_channels, depth_resolution, height / 4, width / 4)
            volume_features = build_volume_features(
                image_features,
                # no need for the reference image
                mvs_images[:, 1:],
                # wlog assume reference matrix is identity
                batch['image_warp_matrices'][:, 1:],
                # use depth bounds for reference frustrum, which we map back to source views
                batch['depth_bounds'][:, 0].squeeze(1),
                padding=self.hparams.image_feature_padding,
                depth_resolution=self.hparams.depth_resolution
            )

            volume_encoding = self.volume_encoding(volume_features)

            # un-normalize images and create ray samples
            mvs_images = self.un_preprocess(mvs_images)
            batch_size, _, channels, height, width = mvs_images.shape

            all_predictions_rgb = torch.empty(
                [batch_size, height, width, channels],
                dtype=mvs_images.dtype,
                device=mvs_images.device,
            )

            batch_row_count = 2
            for image_row in range(0, height, batch_row_count):
                ray_count, ray_sample_count = width * batch_row_count, self.hparams.ray_march_sample_count
                ray_offset_function = create_ray_offsets_row(
                    width,
                    image_row,
                    batch_row_count,
                    batch_size=batch_size,
                    dtype=mvs_images.dtype,
                    device=mvs_images.device
                )
                all_ray_directions, all_ray_origins, all_depth_samples, all_point_samples, all_point_samples_ndc, _ = march_rays(
                    mvs_images,
                    batch['intrinsic_param_matrices'],
                    batch['camera_to_world_matrices'],
                    batch['world_to_camera_matrices'],
                    ray_offset_function,
                    ray_count,
                    ray_sample_count,
                    batch['depth_bounds'],
                    padding=self.hparams.image_feature_padding
                )

                volume_directions = create_direction_vectors(
                    all_ray_directions[:, 0],
                    batch['world_to_camera_matrices'][:, 1],
                    ray_sample_count
                )
                # volume_features.shape: (batch_size, ray_count, ray_sample_count, 8)
                volume_features = interpolate_volume_encoding(
                    volume_encoding,
                    all_point_samples_ndc
                )
                # source_image_colours.shape: (batch_size, channels * source_viewpoints, ray_count, ray_sample_count)
                source_image_colours = interpolate_pixel_colours(
                    all_point_samples[:, 0],
                    mvs_images[:, 1:],
                    batch['world_to_camera_matrices'][:, 1:],
                    batch['intrinsic_param_matrices'][:, 1:],
                    batch['depth_bounds'][:, 1:],
                )
                prediction_colours, prediction_density = self.self_attention_renderer(
                    all_point_samples_ndc.view(batch_size * ray_count * ray_sample_count, 3),
                    volume_directions,
                    volume_features,
                    source_image_colours
                )

                # prediction_rgb.shape: (batch_size, ray_count, channels)
                all_predictions_rgb[:, image_row:image_row + batch_row_count] = parse_nerf(
                    prediction_colours.view(batch_size, ray_count, ray_sample_count, 3),
                    prediction_density.view(batch_size, ray_count, ray_sample_count)
                ) \
                    .view(batch_row_count, width, channels)

            mvs_images_reference = mvs_images[:, 0]
            image_prediction = all_predictions_rgb.permute(0, 3, 1, 2)

            # compute loss and signal-to-noise
            mean_squared_error = self.loss(image_prediction, mvs_images_reference)
            peak_signal_to_noise = self.peak_signal_to_noise(image_prediction, mvs_images_reference)

            self.log('validate/mean_squared_error', mean_squared_error, prog_bar=True)
            self.log('validate/peak_signal_to_noise', peak_signal_to_noise, prog_bar=True)

            # write images
            real = mvs_images_reference[0].permute(1, 2, 0).cpu().numpy()* 255
            pred = image_prediction[0].permute(1, 2, 0).cpu().numpy() * 255

            if random.random() > 0.98:
                cv2.imwrite(f'.experiments/tmp/{self.global_step:08d}_{batch_idx:02d}_real.png', real.astype('uint8'))
                cv2.imwrite(f'.experiments/tmp/{self.global_step:08d}_{batch_idx:02d}_pred.png', pred.astype('uint8'))
            # mvs_images.shape: (batch_size, viewpoints, channels, height, width)
