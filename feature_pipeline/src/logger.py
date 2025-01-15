import logging
from .config import DirectoriesConfig

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    file_handler = logging.FileHandler(
        DirectoriesConfig.FEATURE_PIPELINE_LOGGING_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


    
    