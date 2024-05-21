import numpy as np


class SourceViews:
    """
    Class representing a scored list of source views. Scoring system is outlined in the
    [MVSNet paper](https://arxiv.org/abs/1804.02505).

    Different selection methods are provided for training and validation steps.
    """

    @staticmethod
    def top_k(source_view_count=3):
        """
        Select the top k source views by score. Used for validation.

        :param source_view_count: Number of source views to select
        :return: Source views
        """
        return lambda: range(source_view_count)

    @staticmethod
    def rand_k_top_n(source_view_count=3, randomise_count=5):
        """
        Choose k random values from the top n scored source images.
        Used for training.

        :param source_view_count: Number of source views to select
        :param randomise_count: Number of source views to consider for selection
        :return: Source views
        """
        return lambda: np.random.permutation(range(randomise_count))[:source_view_count]

    def __init__(self, source_views, selection_type):
        """
        Construct a new source views object; given a list of source views and a selection method.

        :param source_views: Source views
        :param selection_type: Selection method `self::top_k` or `self::rand_k_top_n`
        """
        super(SourceViews).__init__()

        self.source_views = source_views
        self.selection_type = selection_type

    def fetch(self):
        """
        Fetch source views using defined selection method.

        :return: Source views
        """
        selection = self.selection_type()

        return [self.source_views[index] for index in selection]

    def fetch_all(self):
        """
        Fetch all source views, without sub-selection.

        :return: All source views
        """
        return self.source_views


def load_image_pairings(file_name, source_view_sampler, expected_source_view_count=10):
    """
    Load image pairings from configuration file.

    :param file_name: Configuration file name
    :param source_view_sampler: Selection method for source views, `SourceViews::top_k` or `SourceViews::rand_n_top_k`.
    :param expected_source_view_count: Expected number of source views in configuration file
    :return: Mapping from target views (all possible viewpoints) to source views
    """
    pairings = dict()
    viewpoint_id_dict = {}

    # parse image pairing configuration file
    with open(file_name, encoding='utf-8') as image_pairing_config:
        # total number of viewpoints, should be 49
        num_viewpoint = int(image_pairing_config.readline())

        for _ in range(num_viewpoint):
            target_view = int(image_pairing_config.readline().rstrip())
            source_views = [int(view_id) for view_id in image_pairing_config.readline().rstrip().split()[1::2]]
            if len(source_views) != expected_source_view_count:
                raise IndexError(
                    'Expected [%d] source views in configuration file, found [%d]' % (
                        expected_source_view_count,
                        len(source_views)
                    )
                )

            pairings[target_view] = SourceViews(source_views, source_view_sampler)

            # mark viewpoints as present in dict
            viewpoint_id_dict[target_view] = True
            for source_view in source_views:
                viewpoint_id_dict[source_view] = True

    return pairings, list(viewpoint_id_dict.keys())
