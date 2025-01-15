import subprocess
from .config import DirectoriesConfig
from . import logger

logger = logger.setup_logger(__name__)


def apply_feast() -> None:
    """
    This function runs 'feast apply' method to register features in the feature store.
    """
    try:
        command = subprocess.run(
            'feast apply', cwd=DirectoriesConfig.FEATURE_STORE_DIR, shell=True, capture_output=True, text=True, check=True)
        
        logger.info(f"command 'feast apply' ran successfully. Output: {command.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(
            f"An error occured while running 'feast apply'. The error was: {e.output}.")
        raise
    except Exception as e:
        logger.error(
            f"An error occured while running 'feast apply'. The error was: {e}.")
        raise
