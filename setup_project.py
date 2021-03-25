from shopee.paths import DATA_ROOT, LOGS_PATH, META_PATH, MODELS_PATH
from shopee.add_meta_csv import create_meta_csv


def create_folders():
    """Helper to create all necessary folders"""
    for p in [DATA_ROOT, LOGS_PATH, MODELS_PATH, META_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def main():
    create_folders()
    create_meta_csv(DATA_ROOT, META_PATH)


if __name__ == "__main__":
    main()
