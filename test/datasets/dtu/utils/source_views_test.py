import pytest

from datasets.dtu.utils.image_pairings import SourceViews


def test_top_k_source_view_count():
    """Test `fetch_top_k` returns top k available source views"""
    top_2_values = [4, 2]
    source_views = SourceViews(top_2_values + [5, 7, 12], SourceViews.top_k(source_view_count=2))

    actual_top_2_values = source_views.fetch()

    assert actual_top_2_values == top_2_values, f'expected top 2 values not found'


def test_top_k_invalid_source_view_count():
    """Test `fetch_top_k` raises error if source view count exceeds available source views"""
    source_views = SourceViews([], SourceViews.top_k(source_view_count=2))

    with pytest.raises(IndexError) as raised_error:
        source_views.fetch()

    # TODO : wrap exception
    assert 'list index out of range' in str(raised_error.value).lower(), 'expected exception not thrown'


def test_top_k_rand_n_source_view_count():
    """Test `fetch_top_k` returns top k available source views"""
    source_views = [4, 2, 5, 7, 12, 21, 11]
    top_5_source_views = source_views[:4]

    source_views = SourceViews(source_views, SourceViews.rand_k_top_n(source_view_count=2, randomise_count=4))
    actual_top_2_values = source_views.fetch()

    assert set(actual_top_2_values) <= set(top_5_source_views), f'source views to contain top 5 elements'


def test_top_k_rand_n_invalid_counts():
    """Test `fetch_rand_k_top_n` raises error if source view count exceeds available source views"""
    source_views = SourceViews([3], SourceViews.rand_k_top_n(source_view_count=2, randomise_count=4))

    with pytest.raises(IndexError) as raised_error:
        source_views.fetch()

    # TODO : wrap exception
    assert 'list index out of range' in str(raised_error.value).lower(), 'expected exception not thrown'
