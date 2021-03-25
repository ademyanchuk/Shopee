import argparse

from shopee.add_meta_csv import create_meta_csv
from shopee.paths import DATA_ROOT, LOGS_PATH, META_PATH, MODELS_PATH

parser = argparse.ArgumentParser(description="Project setup")
parser.add_argument(
    "--build-folds",
    action="store_true",
    help="Create train csv file with folds and meta-info",
)
args = parser.parse_args()


def create_folders():
    """Helper to create all necessary folders"""
    for p in [DATA_ROOT, LOGS_PATH, MODELS_PATH, META_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def main(args):
    create_folders()
    if args.build_folds:
        create_meta_csv(DATA_ROOT, META_PATH)


if __name__ == "__main__":
    main(args)
