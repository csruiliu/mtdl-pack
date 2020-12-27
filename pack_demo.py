import argparse

from pack.core.train_single import train_single
from pack.core.train_pack import train_pack
from pack.core.train_parallel import train_parallel
from pack.core.train_sequential import train_sequential


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pattern', action='store', type=str,
                        choices=['single', 'seq', 'pack', 'parallel'],
                        help='the training pattern')
    args = parser.parse_args()
    train_pattern = args.pattern

    if train_pattern == 'single':
        train_single()
    elif train_pattern == 'seq':
        train_sequential()
    elif train_pattern == 'parallel':
        train_parallel()
    elif train_pattern == 'pack':
        train_pack()
    else:
        raise ValueError('the training pattern is not recognized')
