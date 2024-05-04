from dataclasses import dataclass
import numpy as np


@dataclass
class ImagePairings:
    pairings: dict

    def iterator(self):
        return iter(self.pairings)

    def fetch(self, reference_id):
        return self.pairings[reference_id]

    def fetch_all_viewpoint_ids(self):
        viewpoint_id_dict = {}

        for reference_view in self.pairings:
            source_views = self.pairings[reference_view]

            viewpoint_id_dict[reference_view] = True
            for source_view in source_views.fetch_all():
                viewpoint_id_dict[source_view] = True

        return list(viewpoint_id_dict.keys())


class SourceViews:
    @staticmethod
    def top_k(source_view_count=3):
        return lambda: range(source_view_count)

    @staticmethod
    def rand_k_top_n(source_view_count=3, randomise_count=5):
        return lambda: np.random.permutation(range(randomise_count))[:source_view_count]

    def __init__(self, source_views, selection_type):
        super(SourceViews).__init__()

        self.source_views = source_views
        self.selection_type = selection_type

    def fetch(self):
        selection = self.selection_type()

        return [self.source_views[index] for index in selection]

    def fetch_all(self):
        return self.source_views


def load_image_pairings(file_name, source_view_sampler, expected_source_view_count=10):
    pairings = dict()

    # parse image pairing configuration file
    with open(file_name, encoding='utf-8') as image_pairing_config:
        # total number of viewpoints, should be 49
        num_viewpoint = int(image_pairing_config.readline())

        for _ in range(num_viewpoint):
            reference_view = int(image_pairing_config.readline().rstrip())
            source_views = [int(view_id) for view_id in image_pairing_config.readline().rstrip().split()[1::2]]
            if len(source_views) != expected_source_view_count:
                raise IndexError(
                    'Expected [%d] source views in configuration file, found [%d]' % (
                        expected_source_view_count,
                        len(source_views)
                    )
                )

            pairings[reference_view] = SourceViews(source_views, source_view_sampler)

    return ImagePairings(pairings=pairings)
