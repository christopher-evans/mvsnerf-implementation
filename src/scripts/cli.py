import argparse

ISSUES_URL = 'https://github.com/christopher-evans/mvsnerf-implementation/issues'

def create_train_arg_parser(name, description):
    parser = argparse.ArgumentParser(
        prog=name,
        description=description,
        epilog=f'Report issues at: {ISSUES_URL}',
    )

    general = parser.add_argument_group('dataset')
    general.add_argument('-n', '--name', type=str, required=True, description='Data set name, one of "dtu"')

    return parser
