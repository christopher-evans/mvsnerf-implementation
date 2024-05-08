import argparse

from scripts.train import train

ISSUES_URL = 'https://github.com/christopher-evans/mvsnerf-implementation/issues'
COMMAND_MAP = {
    'train': train
}


def create_train_parser(actions):
    train = actions.add_parser('train', help='Train MVSNeRF network')
    train_dataset = train.add_argument_group('dataset')
    train_dataset.add_argument(
        '--dataset',
        type=str,
        required=True,
        default='dtu',
        help='Data set name, one of "dtu"'
    )
    train_dataset.add_argument(
        '--data_dir',
        type=str,
        required=False,
        help='Location of dataset'
    )
    train_dataset.add_argument(
        '--data_config_dir',
        type=str,
        required=False,
        help='Location of dataset configuration'
    )
    train_dataset.add_argument(
        '--data_max_length',
        type=int,
        default=-1,
        help='Maximum number of images to load from the dataset'
    )

    train_dataset = train.add_argument_group('hyperparameters')
    train_dataset.add_argument(
        '--learning_rate',
        type=float,
        default=5e-4,
        help='Learning rate used for training'
    )

def create_mvsnerf_parser():
    # create top level command
    command = argparse.ArgumentParser(
        prog='MVSNeRF',
        description='Command line tool for training and evaluating MVSNeRF network',
        epilog=f'Report issues at: {ISSUES_URL}',
    )
    actions = command.add_subparsers(dest='command')

    # add commands
    create_train_parser(actions)
    #create_validate_parser(actions)
    #create_fine_tune_parser(actions)
    #create_infer_parser(actions)

    return command

if __name__ == '__main__':
    arg_parser = create_mvsnerf_parser()
    args = arg_parser.parse_args()
    command = args.command

    action = COMMAND_MAP[args.command]
    action(args)