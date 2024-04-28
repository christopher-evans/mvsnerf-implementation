import pytest
import test.data
import numpy as np

from utils.pfm import read_pfm_file
from importlib.resources import files


@pytest.fixture
def pfm_data_path():
    return files(test.data).joinpath('pfm')


def test_invalid_pfm_header(pfm_data_path):
    with pytest.raises(IOError) as excinfo:
        read_pfm_file(pfm_data_path.joinpath('invalid-pfm-declaration.txt'))

    assert 'not a pfm file' in str(excinfo.value).lower()


def test_invalid_pfm_dims(pfm_data_path):
    with pytest.raises(IOError) as excinfo:
        read_pfm_file(pfm_data_path.joinpath('invalid-pfm-header.txt'))

    assert 'malformed pfm header' in str(excinfo.value).lower()


def test_valid_pfm(pfm_data_path):
    pfm_data, scale = read_pfm_file(pfm_data_path.joinpath('valid.pfm'))
    expected_pfm_data = np.array([
        [
            [2.0, 2.0, 2.0],
            [2.0, 3.0, 2.0],
            [2.0, 4.0, 2.0]
        ],
        [
            [1.0, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 4.0, 1.0]
        ],
    ]).astype(np.float32)

    np.testing.assert_array_almost_equal(
        pfm_data,
        expected_pfm_data,
        decimal=6,
        err_msg='Failed to match little endian PFM data'
    )