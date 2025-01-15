from typing import  Dict, Any
import torch
from .logger import setup_logger
from .constants import (
    EXPERIMENT_CONFIG_REQUIRED_KEYS,
    HyperparameterTuningMethods,
    Models,
    Preprocessors
)

logger = setup_logger(__name__)

def check_gpu() -> None:
    if torch.cuda.is_available():
        logger.info("GPU is available.")
    else:
        logger.warning("GPU is not available. Training will be done on CPU.")

def validate_keys(config: Dict[str, Any]) -> None:
    missing_keys = [key for key in EXPERIMENT_CONFIG_REQUIRED_KEYS if key not in config]
    error_message = f"Missing keys in experiment config: {missing_keys}"
    if missing_keys:
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info("All keys are present in the experiment config.")
    
def validate_tuning_method(tuning_method: str) -> None:
    if tuning_method not in HyperparameterTuningMethods.list():
        error_message = f"Invalid tuning method: {tuning_method}. Must be one of {HyperparameterTuningMethods.list()}."
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"Tuning method: {tuning_method} is valid.")
    
def validate_model(model_name: str) -> None:
    if model_name not in Models.list():
        error_message = f"Invalid model name: {model_name}. Must be one of {Models.list()}."
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"Model name: {model_name} is valid.")
    
def validate_preprocessor(preprocessor_method: str) -> None:
    if preprocessor_method not in Preprocessors.list():
        error_message = f"Invalid preprocessor method: {preprocessor_method}. Must be one of {Preprocessors.list()}."
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"Preprocessor method: {preprocessor_method} is valid.")
    
    
def validate_experiment_config(experiment_config: Dict[str, Any]) -> None:

    validate_keys(experiment_config)
    if experiment_config['model_name'] == 'distilbert':
        check_gpu()
    if experiment_config['train_with_tuning']:
        validate_tuning_method(experiment_config['tuning_method'])

    validate_model(experiment_config['model_name'])

    validate_preprocessor(experiment_config['preprocessor_method'])
    
