import argparse
from dataclasses import dataclass
from scripts.utils import Files


def main(arguments):
    files = Files(arguments.dataset_path)
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',
                        default='data/dataset/celeba',
                        type=str,
                        help='Dataset path'
                        )
    parser.add_argument('--supervised_fraction',
                        default=0.2,
                        type=float,
                        help='Fraction of supervised data'
                        )
    args = parser.parse_args()

    main(args)
