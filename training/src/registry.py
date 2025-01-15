from .constants import Models, Preprocessors, HyperparameterTuningMethods
from .models import LogisticRegressionModel, RandomForestModel, XgboostModel
from .configs import LlamaThreeSettings
from .preprocessor import StandardScalerPreprocessor
from .hp_tuner import RandomSearchTuner, HyperOptTuner
from .logger import setup_logger
from typing import Dict, Any

logger = setup_logger(__name__)

class RegisterModel:
    
    SUPPORTED_MODELS = {
        Models.LOGISTIC_REGRESSION : LogisticRegressionModel(),
        Models.RANDOM_FOREST : RandomForestModel(),
        Models.XGBOOST : XgboostModel(),
    }
    
    @staticmethod
    def instantiate_model(model_name: str, configs: Dict[str, Any]):
        if model_name not in RegisterModel.SUPPORTED_MODELS:
            error_message = f"Model {model_name} is not available."
            logger.error(error_message)
            raise KeyError(error_message)
        else:                
            model = RegisterModel.SUPPORTED_MODELS[model_name]
            logger.info(f'Model {model_name} is registered.')
            return model.initialize_model(configs)
               
class RegisterPreprocessor:

    SUPPORTED_PREPROCESSORS = {
        Preprocessors.STANDARD_SCALER : StandardScalerPreprocessor(),  
    }
    
    @staticmethod
    def instantiate_preprocessor(preprocessor_method: str):
        if preprocessor_method not in RegisterPreprocessor.SUPPORTED_PREPROCESSORS:
            error_message = f"Preprocessor {preprocessor_method} is not available."
            logger.error(error_message)
            raise KeyError(error_message)
        else:                
            preprocessor = RegisterPreprocessor.SUPPORTED_PREPROCESSORS[preprocessor_method]
            logger.info(f'Preprocessor {preprocessor_method} is registered.')
            return preprocessor.create_preprocessor_strategy()
    

class RegisterHpTuner:
    SUPPORTED_HP_TUNERS = {
        HyperparameterTuningMethods.RANDOM_SEARCH : RandomSearchTuner(),
        HyperparameterTuningMethods.HYPEROPT : HyperOptTuner()
    }
    
    @staticmethod
    def instantiate_tuner(hyperparameter_tuning_method: str, *args, **kwargs):
        if hyperparameter_tuning_method not in RegisterHpTuner.SUPPORTED_HP_TUNERS:
            error_message = f"Hyperparameter tuner {hyperparameter_tuning_method} is not available."
            logger.error(error_message)
            raise KeyError(error_message)
        else:                
            hp_tuner = RegisterHpTuner.SUPPORTED_HP_TUNERS[hyperparameter_tuning_method]
            logger.info(f'Hyperparameter tuner {hyperparameter_tuning_method} is registered.')
            return hp_tuner.create_tuning_strategy(*args, **kwargs)
        
class RegistryLlmModels:
    SUPPORTED_LLMs = {
    Models.LLAMA: LlamaThreeSettings,
}
    @staticmethod
    def get_model_settings(llm_name: str):
        if llm_name not in RegistryLlmModels.SUPPORTED_LLMs:
            error_message = f"Model {llm_name} is not available."
            logger.error(error_message)
            raise KeyError(error_message)
        return RegistryLlmModels.SUPPORTED_LLMs[llm_name]
    
