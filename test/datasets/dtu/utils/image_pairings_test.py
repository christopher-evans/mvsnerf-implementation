import pytest
import test.config

from datasets.dtu.utils.image_pairings import load_image_pairings, SourceViews
from importlib.resources import files


@pytest.fixture
def dtu_config_path():
    return files(test.config).joinpath('dtu_example/split_example')


def test_image_pairings_valid_file(dtu_config_path):
    """Test `fetch` for valid image pairing file"""
    test_file_name = dtu_config_path.joinpath('sample_image_pairing.txt')
    source_view_sampler = SourceViews.top_k(source_view_count=3)
    image_pairings, all_viewpoint_ids = load_image_pairings(test_file_name, source_view_sampler=source_view_sampler)

    # check a few examples from file
    assert image_pairings[0].fetch() == [10, 1, 9]

    assert image_pairings[1].fetch() == [9, 10, 2]
    assert image_pairings[5].fetch() == [6, 18, 4]

    # all viewpoints present
    assert len(all_viewpoint_ids) == 49


def test_image_pairings_invalid_file_format(dtu_config_path):
    """Test parsing fails for invalid image pairing file format"""
    test_file_name = dtu_config_path.joinpath('sample_invalid_image_pairing.txt')
    with pytest.raises(ValueError) as raised_error:
        load_image_pairings(test_file_name, source_view_sampler=SourceViews.top_k())

    assert 'invalid literal for int' in str(raised_error.value).lower(), 'expected exception not thrown'


def test_image_pairings_invalid_source_view_count(dtu_config_path):
    """Test `fetch` raises error if expected source view count does not match number found in file"""
    expected_source_view_count = 11
    test_file_name = dtu_config_path.joinpath('sample_image_pairing.txt')

    with pytest.raises(IndexError) as raised_error:
        load_image_pairings(
            test_file_name,
            source_view_sampler=SourceViews.top_k(),
            expected_source_view_count=expected_source_view_count
        )

    assert 'source views in configuration file' in str(raised_error.value).lower(), 'expected exception not thrown'
    assert '[11]' in str(raised_error.value).lower(), 'expected exception not thrown'
