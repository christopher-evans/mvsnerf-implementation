import torch
import pytest

from utils.cost_volume import create_source_depth_values


def test_create_reference_depth_values():
    """Test `create_source_depth_values` with no translation returns all zeros"""
    batch_size = 2
    image_height = 2
    image_width = 3
    depth_resolution = 3
    depth_bounds = torch.tensor([
        [1, 2],
        [3, 4]
    ])
    source_translation = torch.tensor([[0, 0, 0], [0, 0, 0]]) \
        .reshape(batch_size, 3, 1)

    source_depth_values = create_source_depth_values(
        depth_bounds,
        image_height,
        image_width,
        depth_resolution,
        source_translation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width)

    assert torch.allclose(source_depth_values, torch.zeros_like(source_depth_values))


@pytest.mark.parametrize(
    ("test_name", "offset_axis", "offset_value", "depth_min", "depth_max"),
    [
        ("x axis, offset 1, depths: [1, 2]", 0, 1, 1, 2),
        ("y axis, offset 1, depths: [1, 2]", 1, 1, 1, 2),
        ("z axis, offset 1, depths: [1, 2]", 2, 1, 1, 2),

        ("x axis, offset 1, depths: [2, 4]", 0, 1, 2, 4),
        ("y axis, offset 1, depths: [2, 4]", 1, 1, 2, 4),
        ("z axis, offset 1, depths: [2, 4]", 1, 1, 2, 4),

        ("x axis, offset 2, depths: [-1, -2]", 0, 2, -1, -2),
        ("y axis, offset 2, depths: [-1, -2]", 1, 2, -1, -2),
        ("z axis, offset 2, depths: [-1, -2]", 2, 2, -1, -2),
    ],
    ids=[
        'x axis, offset 1, depths: [1, 2]',
        'y axis, offset 1, depths: [1, 2]',
        'z axis, offset 1, depths: [1, 2]',

        'x axis, offset 1, depths: [2, 4]',
        'y axis, offset 1, depths: [2, 4]',
        'z axis, offset 1, depths: [2, 4]',

        'x axis, offset 2, depths: [-1, -2]',
        'y axis, offset 2, depths: [-1, -2]',
        'z axis, offset 2, depths: [-1, -2]',
    ]
)
def test_create_source_depth_values_single_offset_batch1(test_name, offset_axis, offset_value, depth_min, depth_max):
    """Test `create_source_depth_values` with offset along one axis with batch size of 1"""
    batch_size = 1
    image_height = 2
    image_width = 3
    depth_resolution = 3
    depth_bounds = torch.tensor([
        [depth_min, depth_max]
    ])
    translation_vector = torch.zeros((1, 3))
    translation_vector[0][offset_axis] = offset_value
    source_translation = translation_vector.reshape(batch_size, 3, 1)

    source_depth_values = create_source_depth_values(
        depth_bounds,
        image_height,
        image_width,
        depth_resolution,
        source_translation
    ) \
        .view(3, depth_resolution, image_height * image_width)

    # verify axis translations
    first_depth = offset_value / depth_min
    mid_depth = offset_value * 2 / (depth_min + depth_max)
    last_depth = offset_value / depth_max

    assert torch.allclose(source_depth_values[offset_axis][0], torch.full((image_height * image_width,), first_depth))
    assert torch.allclose(source_depth_values[offset_axis][1], torch.full((image_height * image_width,), mid_depth))
    assert torch.allclose(source_depth_values[offset_axis][2], torch.full((image_height * image_width,), last_depth))

    # remaining axis translations are all zero
    zero_axes = torch.cat((source_depth_values[:offset_axis], source_depth_values[offset_axis + 1:]))
    assert torch.allclose(zero_axes, torch.zeros_like(zero_axes))


def test_create_source_depth_values_datatype():
    """Test `create_source_depth_values` returns a tensor with the same datatype as the depth bounds"""
    batch_size = 1
    image_height = 2
    image_width = 3
    depth_resolution = 3
    source_data_type = torch.float16
    depth_bounds = torch.tensor([
        [1.0, 2.2]
    ], dtype=source_data_type)
    source_translation = torch.tensor([[0, 0, 0]], dtype=source_data_type) \
        .reshape(batch_size, 3, 1)

    source_depth_values = create_source_depth_values(
        depth_bounds,
        image_height,
        image_width,
        depth_resolution,
        source_translation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width)

    assert source_depth_values.dtype == source_data_type


def test_create_source_depth_values_multiple_offset_batch2():
    """Test `create_source_depth_values` with offset along one axis with batch size of 1"""
    batch_size = 2

    image_height = 2
    image_width = 3
    depth_resolution = 3

    depth_min_first_batch = 1
    depth_max_first_batch = 2
    depth_min_second_batch = 2
    depth_max_second_batch = 4
    depth_bounds = torch.tensor([
        [depth_min_first_batch, depth_max_first_batch],
        [depth_min_second_batch, depth_max_second_batch],
    ])

    offset_first_batch = 1
    offset_second_batch = -1
    source_translation = torch.tensor([
        [offset_first_batch, offset_first_batch, offset_first_batch],
        [offset_second_batch, offset_second_batch, offset_second_batch],
    ], dtype=torch.float32) \
        .reshape(batch_size, 3, 1)

    source_depth_values = create_source_depth_values(
        depth_bounds,
        image_height,
        image_width,
        depth_resolution,
        source_translation
    ) \
        .view(batch_size, 3, depth_resolution, image_height * image_width)

    # first batch, check depths for each axis
    first_depth = offset_first_batch / depth_min_first_batch
    mid_depth = offset_first_batch * 2 / (depth_min_first_batch + depth_max_first_batch)
    last_depth = offset_first_batch / depth_max_first_batch

    first_batch_depths = source_depth_values[0]
    assert torch.allclose(first_batch_depths[:, 0], torch.full((3, image_height * image_width,), first_depth))
    assert torch.allclose(first_batch_depths[:, 1], torch.full((3, image_height * image_width,), mid_depth))
    assert torch.allclose(first_batch_depths[:, 2], torch.full((3, image_height * image_width,), last_depth))

    # second batch, check depths for each axis
    first_depth = offset_second_batch / depth_min_second_batch
    mid_depth = offset_second_batch * 2 / (depth_min_second_batch + depth_max_second_batch)
    last_depth = offset_second_batch / depth_max_second_batch

    second_batch_depths = source_depth_values[1]
    assert torch.allclose(second_batch_depths[:, 0], torch.full((3, image_height * image_width,), first_depth))
    assert torch.allclose(second_batch_depths[:, 1], torch.full((3, image_height * image_width,), mid_depth))
    assert torch.allclose(second_batch_depths[:, 2], torch.full((3, image_height * image_width,), last_depth))
