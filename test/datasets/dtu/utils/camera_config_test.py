import pytest
import test.data
import numpy as np

from datasets.dtu.utils.camera_config import load_camera_matrices, CameraMatrices
from importlib.resources import files


@pytest.fixture
def dtu_data_path():
    return files(test.data).joinpath('dtu_example')


@pytest.fixture
def test_camera_matrices():
    return CameraMatrices(
        world_to_camera=np.array([
            [ 0.970263  ,  0.00747983,  0.241939  , -0.9551    ],
            [-0.0147429 ,  0.999493  ,  0.0282234 ,  0.0164416 ],
            [-0.241605  , -0.030951  ,  0.969881  ,  0.1127005 ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ]),
        camera_to_world=np.array([
            [ 0.97026235, -0.01474279, -0.2416051 ,  0.954169  ],
            [ 0.00747994,  0.9994928 , -0.03095099, -0.00580098],
            [ 0.2419387 ,  0.02822343,  0.9698809 ,  0.12130555],
            [ 0.        ,  0.        ,  0.        ,  1.        ]
        ]),
        intrinsic_params=np.array([
            [1.446165e+03, 0.000000e+00, 3.316025e+02],
            [0.000000e+00, 1.441590e+03, 2.655355e+02],
            [0.000000e+00, 0.000000e+00, 1.000000e+00]
        ]),
        projection_matrix=np.array([
            [ 3.30760895e+02,  1.38409898e-01,  1.67874680e+02, -3.35965088e+02],
            [-2.13519802e+01,  3.58160126e+02,  7.45560989e+01,  1.34070072e+01],
            [-2.41604999e-01, -3.09510008e-02,  9.69880998e-01,  1.12700500e-01],
            [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
        ]),
        depth_bounds=(2.125, 4.525)
    )

def test_image_pairings_valid_file(dtu_data_path, test_camera_matrices):
    """Test `fetch` for valid image pairing file"""
    test_viewpoint_id = 0
    test_scale_factor = 1.0 / 200
    test_down_sample = 1.0
    actual_camera_matrices = load_camera_matrices(
        dtu_data_path,
        test_viewpoint_id,
        scale_factor=test_scale_factor,
        down_sample=test_down_sample
    )

    assert np.allclose(
        actual_camera_matrices.world_to_camera,
        test_camera_matrices.world_to_camera
    ), 'Expected world to camera matrices to match'
    assert np.allclose(
        np.matmul(actual_camera_matrices.camera_to_world, actual_camera_matrices.world_to_camera),
        np.eye(4),
        atol=1e-7
    ), 'Expected world to camera to be inverse of camera to world matrix'
    assert np.allclose(
        actual_camera_matrices.intrinsic_params,
        test_camera_matrices.intrinsic_params
    ), 'Expected intrinsic parameters to match'
    assert np.allclose(
        actual_camera_matrices.projection_matrix,
        test_camera_matrices.projection_matrix
    ), 'Expected projection matrices to match'
    assert np.allclose(
        actual_camera_matrices.depth_bounds,
        test_camera_matrices.depth_bounds
    ), 'Expected depth bounds to match'
    assert np.allclose(
        actual_camera_matrices.world_to_camera,
        test_camera_matrices.world_to_camera
    ), 'Expected world to camera matrices to match'