import pytest
import test.config

from lightning.pytorch.trainer.states import TrainerFn
from datasets.dtu.utils.scan_config import load_scans
from importlib.resources import files


@pytest.fixture
def dtu_config_path():
    return files(test.config).joinpath('dtu_example/split_example')


def test_scan_config(dtu_config_path):
    """Test `fetch_scans` for valid scan config file"""
    scans = load_scans(dtu_config_path, TrainerFn.FITTING)

    # check a few examples from file
    assert scans == ['scan114', 'scan115', 'scan1']


def test_scan_config_newline(dtu_config_path):
    """Test `fetch_scans` for valid scan config file ending in new line"""
    scans = load_scans(dtu_config_path, TrainerFn.VALIDATING)

    # check a few examples from file
    assert scans == ['scan114', 'scan115', 'scan1']
