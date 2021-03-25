from shopee.paths import DATA_ROOT, LOGS_PATH, META_PATH, MODELS_PATH


def create_folders():
    """Helper to create all necessary folders"""
    for p in [DATA_ROOT, LOGS_PATH, MODELS_PATH, META_PATH]:
        p.mkdir(parents=True, exist_ok=True)


def main():
    create_folders()


if __name__ == "__main__":
    main()
