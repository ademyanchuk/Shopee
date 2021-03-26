import logging
import subprocess
from datetime import datetime
from pathlib import Path


def get_exp_name(debug: bool):
    now = datetime.now()
    if debug:
        exp_name = "debug_" + now.strftime("%d-%m-%Y-%H-%M")
    else:
        exp_name = "exp_" + now.strftime("%d-%m-%Y-%H-%M")
    return exp_name


def get_commit_hash() -> str:
    commit_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
    return commit_hash.decode()


def setup_logg(
    logs_path: Path, exp_name: str, debug: bool = False
) -> logging.RootLogger:
    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    log_format = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    logger.handlers = [stream_handler]
    # add logging handler to save logs to the file
    log_fname = f"{logs_path}/{exp_name}.log"
    file_handler = logging.FileHandler(log_fname, mode="a")
    file_handler.setFormatter(log_format)
    logger.handlers.append(file_handler)
    return logger
