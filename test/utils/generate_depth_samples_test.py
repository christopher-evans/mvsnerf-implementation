import torch
import pytest

from utils.ray_marching import generate_depth_samples


def test_generate_depth_samples_data_type():
    """Test `generate_depth_samples` returns tensor with correct data type"""
    depth_min = 1
    depth_max = 2
    depth_bounds = torch.tensor([[depth_min, depth_max]], dtype=torch.float16)
    ray_count = 3
    ray_sample_count = 2

    depth_samples = generate_depth_samples(depth_bounds, ray_count, ray_sample_count)
    assert depth_samples.dtype == depth_bounds.dtype, 'data types match'


@pytest.mark.parametrize(
    ("test_name", "depth_min", "depth_max"),
    [
        ("depths: [0, 1]", 0, 1),
        ("depths: [1, 2]", 1, 2),
        ("depths: [1, 12]", 1, 12),
        ("depths: [-1, -12]", -1, -12),
    ],
    ids=[
        'single batch two samples, depths: [0, 1]',
        'single batch two samples, depths: [1, 2]',
        'single batch two samples, depths: [1, 12]',
        'single batch two samples, depths: [-1, -12]',
    ]
)
def test_generate_depth_samples_single(test_name, depth_min, depth_max):
    """Test `generate_depth_samples` for batch size 1"""
    depth_bounds = torch.tensor([[depth_min, depth_max]], dtype=torch.float32)
    ray_count = 3
    ray_sample_count = 2

    # use generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(0)
    depth_samples = generate_depth_samples(depth_bounds, ray_count, ray_sample_count, generator=generator)

    # reset generator seed
    generator.manual_seed(0)
    sample_noise = torch.rand((1, ray_count, ray_sample_count), generator=generator)

    depth_diff = (depth_max - depth_min) / 2
    assert torch.allclose(depth_samples[0, :, 0], depth_min + sample_noise[0, :, 0] * depth_diff), 'first sample points match'
    assert torch.allclose(depth_samples[0, :, 1], depth_min + depth_diff * (sample_noise[0, :, 1] + 1)), 'last sample points match'


@pytest.mark.parametrize(
    ("test_name", "first_depth_min", "first_depth_max", "second_depth_min", "second_depth_max"),
    [
        ("depths: [[0, 1], [1, 2]]", 0, 1, 1, 2),
        ("depths: [[1, 12], [-1, -12]]", 1, 12, -1, -12),
    ],
    ids=[
        'single batch two samples, depths: [[0, 1], [1, 2]]',
        'single batch two samples, depths: [[1, 12], [-1, -12]]'
    ]
)
def test_generate_depth_samples_batch(test_name, first_depth_min, first_depth_max, second_depth_min, second_depth_max):
    """Test `generate_depth_samples` for batch size 2"""
    depth_bounds = torch.tensor([
        [first_depth_min, first_depth_max],
        [second_depth_min, second_depth_max]
    ], dtype=torch.float32)
    ray_count = 3
    ray_sample_count = 2
    batch_size = 2

    # use generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(0)
    depth_samples = generate_depth_samples(depth_bounds, ray_count, ray_sample_count, generator=generator)

    # reset generator seed
    generator.manual_seed(0)
    sample_noise = torch.rand((batch_size, ray_count, ray_sample_count), generator=generator)

    first_depth_diff = (first_depth_max - first_depth_min) / 2
    assert torch.allclose(depth_samples[0, :, 0], first_depth_min + sample_noise[0, :, 0] * first_depth_diff), 'first sample points match first batch item'
    assert torch.allclose(depth_samples[0, :, 1], first_depth_min + first_depth_diff * (sample_noise[0, :, 1] + 1)), 'last sample points match first batch item'

    second_depth_diff = (second_depth_max - second_depth_min) / 2
    assert torch.allclose(depth_samples[1, :, 0], second_depth_min + sample_noise[1, :, 0] * second_depth_diff), 'first sample points match second batch item'
    assert torch.allclose(depth_samples[1, :, 1], second_depth_min + second_depth_diff * (sample_noise[1, :, 1] + 1)), 'last sample points match second batch item'
