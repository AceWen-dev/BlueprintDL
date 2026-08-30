import logging
import os


def get_logger(work_dir=None, name='dlkit'):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(work_dir, 'log.txt'), encoding='utf-8'
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger
