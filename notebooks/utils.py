import logging
import os

import colorlog
import time


def logger():
    logger = logging.getLogger()
    if len(logger.handlers) > 0:
        return logger
    log_colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red,bg_white'
    }
    formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
        log_colors=log_colors
    )
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def timestamp():
    return int(time.time())


def file_with_timestamp(name: str, tmstp=None):
    extension_index = name.rfind('.')
    if tmstp is None:
        return name[:extension_index] + '-' + str(timestamp()) + name[extension_index:]
    else:
        return name[:extension_index] + '-' + str(tmstp) + name[extension_index:]


def latest_file(folder: str, file_name=None):
    files = os.listdir(folder)
    files.sort(reverse=True)
    if file_name is not None:
        files = [f for f in files if f.split('-')[0] == file_name]
    return os.path.join(folder, files[0])

