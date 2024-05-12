import torch

from utils.cost_volume import create_source_plane_values


def test_create_source_plane_values_datatype():
    """Test `create_source_plane_values` returns a tensor with the same datatype as the source rotation"""
    batch_size = 1
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3
    source_data_type = torch.float16
    source_rotation = torch.eye(3, dtype=source_data_type).unsqueeze(0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    )

    assert source_plane_values.dtype == source_data_type


def test_create_reference_plane_values():
    """Test `create_source_plane_values` with identity rotation returns padded reference coordinates"""
    batch_size = 1
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3
    source_rotation = torch.eye(3).unsqueeze(0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width) \
        .squeeze()

    expected_values = torch.tensor([
        # x values
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        # y values
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        # z values
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(source_plane_values[:, 0], expected_values)
    assert torch.allclose(source_plane_values[:, 1], expected_values)
    assert torch.allclose(source_plane_values[:, 2], expected_values)


def test_create_source_plane_values_flip_x():
    """Test `create_source_plane_values` with rotation which flips about x axis"""
    batch_size = 1
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3

    source_rotation = torch.eye(3)
    source_rotation[0][0] = -1
    source_rotation = source_rotation.unsqueeze(0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width) \
        .squeeze()

    expected_values = torch.tensor([
        # x values
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ],
        # y values
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        # z values
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(source_plane_values[:, 0], expected_values)
    assert torch.allclose(source_plane_values[:, 1], expected_values)
    assert torch.allclose(source_plane_values[:, 2], expected_values)


def test_create_source_plane_values_flip_y():
    """Test `create_source_plane_values` with rotation which flips about y axis"""
    batch_size = 1
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3

    source_rotation = torch.eye(3)
    source_rotation[1][1] = -1
    source_rotation = source_rotation.unsqueeze(0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width) \
        .squeeze()

    expected_values = torch.tensor([
        # x values
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        # y values
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ],
        # z values
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(source_plane_values[:, 0], expected_values)
    assert torch.allclose(source_plane_values[:, 1], expected_values)
    assert torch.allclose(source_plane_values[:, 2], expected_values)


def test_create_source_plane_values_flip_z():
    """Test `create_source_plane_values` with rotation which flips about z axis"""
    batch_size = 1
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3

    source_rotation = torch.eye(3)
    source_rotation[2][2] = -1
    source_rotation = source_rotation.unsqueeze(0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width) \
        .squeeze()

    expected_values = torch.tensor([
        # x values
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        # y values
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        # z values
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(source_plane_values[:, 0], expected_values)
    assert torch.allclose(source_plane_values[:, 1], expected_values)
    assert torch.allclose(source_plane_values[:, 2], expected_values)



def test_create_source_plane_values_batches():
    """Test `create_source_plane_values` with batch size greater than 1"""
    batch_size = 2
    padding = 1
    image_height = 3
    image_width = 3
    depth_resolution = 3

    first_batch_rotation = torch.eye(3)
    first_batch_rotation[0][0] = -1
    second_batch_rotation = torch.eye(3)
    second_batch_rotation[2][2] = -1

    source_rotation = torch.cat([first_batch_rotation.unsqueeze(0), second_batch_rotation.unsqueeze(0)], dim=0)

    source_plane_values = create_source_plane_values(
        image_height,
        image_width,
        padding,
        depth_resolution,
        batch_size,
        source_rotation
    ) \
        .view(batch_size, 3, depth_resolution, image_height, image_width)

    expected_value_first_batch = torch.tensor([
        # x values
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ],
        # y values
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        # z values
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
    ], dtype=torch.float32)
    expected_value_second_batch = torch.tensor([
        # x values
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        # y values
        [
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ],
        # z values
        [
            [-1, -1, -1],
            [-1, -1, -1],
            [-1, -1, -1]
        ]
    ], dtype=torch.float32)

    assert torch.allclose(source_plane_values[0, :, 0], expected_value_first_batch)
    assert torch.allclose(source_plane_values[0, :, 1], expected_value_first_batch)
    assert torch.allclose(source_plane_values[0, :, 2], expected_value_first_batch)

    assert torch.allclose(source_plane_values[1, :, 0], expected_value_second_batch)
    assert torch.allclose(source_plane_values[1, :, 1], expected_value_second_batch)
    assert torch.allclose(source_plane_values[1, :, 2], expected_value_second_batch)
