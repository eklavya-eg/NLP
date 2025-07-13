import logging
import datetime
import torch

def mkLogger(name:str):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # Set the logging level to control which messages get logged
    file_handler = logging.FileHandler(name)
    file_handler.setLevel(logging.DEBUG)  # Set the level for this handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the level for this handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
