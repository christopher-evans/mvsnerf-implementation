import argparse
import sys

from scripts.train import train as train_script


ISSUES_URL = 'https://github.com/christopher-evans/mvsnerf-implementation/issues'
COMMAND_MAP = {
    'train': train_script
}


def create_train_parser(actions):
    train = actions.add_parser('train', help='Train MVSNeRF network')
    train.add_argument(
        '--experiment_name',
        type=str,
        default='mvsnerf',
        help='Name to display in tensorboard for experiment'
    )

    train_dataset = train.add_argument_group('dataset')
    train_dataset.add_argument(
        '--dataset',
        type=str,
        default='dtu',
        help='Data set name, one of "dtu"'
    )
    train_dataset.add_argument(
        '--data_dir',
        type=str,
        default='.data/processed/dtu_example',
        help='Location of dataset'
    )
    train_dataset.add_argument(
        '--data_config_dir',
        type=str,
        default='.configs/dtu_example/split_example',
        help='Location of dataset configuration'
    )
    train_dataset.add_argument(
        '--data_max_length',
        type=int,
        default=-1,
        help='Maximum number of images to load from the dataset'
    )
    train_dataset.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Maximum number of images to load from the dataset'
    )
    train_dataset.add_argument(
        '--depth_scale_factor',
        type=float,
        default=1.0 / 200,
        help='Scale factor for depth z -> z * depth_scale_factor'
    )
    train_dataset.add_argument(
        '--image_down_sample',
        type=float,
        default=1.0,
        help='Down sample dataset images (x, y) -> (x * down_sample, y * down_sample)'
    )

    train_hyperparameters = train.add_argument_group('hyperparameters')
    train_hyperparameters.add_argument(
        '--initial_learning_rate',
        type=float,
        default=5e-4,
        help='Initial learning rate used for training'
    )
    train_hyperparameters.add_argument(
        '--minimum_learning_rate',
        type=float,
        default=1e-7,
        help='Minimum learning rate used by scheduler'
    )
    train_hyperparameters.add_argument(
        '--epoch_count',
        type=int,
        default=6,
        help='Number of epochs for training'
    )

    train_rendering = train.add_argument_group('rendering')
    train_rendering.add_argument(
        '--image_feature_padding',
        type=int,
        default=24,
        help='Padding for image features when evaluating the warped features to produce a cost volume'
    )
    train_rendering.add_argument(
        '--depth_resolution',
        type=int,
        default=128,
        help='Number of depth increments used for the cost volume'
    )
    train_rendering.add_argument(
        '--ray_direction_random_sampling',
        type=bool,
        default=True,
        help='If True, sample pixel offsets for ray directions at random; if false use all possible pixel offsets'
    )
    train_rendering.add_argument(
        '--ray_march_count',
        type=int,
        default=1024,
        help='If ray_direction_random_sampling is set, the number of samples to use'
    )
    train_rendering.add_argument(
        '--ray_march_sample_count',
        type=int,
        default=128,
        help='Number of depth samples to take along each ray'
    )

    train_rendering = train.add_argument_group('debugging')
    train_rendering.add_argument(
        '--limit_train_batches',
        type=int,
        default=None,
        help='Limit the number of batches processed each epoch'
    )


def create_mvsnerf_parser():
    # create top level command
    arg_parser = argparse.ArgumentParser(
        prog='MVSNeRF',
        description='Command line tool for training and evaluating MVSNeRF network',
        epilog=f'Report issues at: {ISSUES_URL}',
    )
    sub_parser = arg_parser.add_subparsers(dest='command')

    # add commands
    create_train_parser(sub_parser)
    #create_validate_parser(actions)
    #create_fine_tune_parser(actions)
    #create_infer_parser(actions)

    return arg_parser


if __name__ == '__main__':
    arg_parser = create_mvsnerf_parser()
    args = arg_parser.parse_args()
    command = args.command
    if command not in COMMAND_MAP:
        arg_parser.print_help()
        sys.exit(2)

    action = COMMAND_MAP[args.command]
    action(args)