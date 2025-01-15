from enum import Enum
from typing import List
from pathlib import Path

# DIRECTORIES 
ROOT_DIR: Path = Path(__file__).parents[2]
FEATURE_STORE_DIR: Path = ROOT_DIR / 'feature_store'/ 'feature_repo'
TRAINING_PIPELINE_LOGGING_DIR: Path = ROOT_DIR / 'training' / 'logs' / 'training_pipeline.log'
LLM_MODELS_DIR: Path = ROOT_DIR / 'training' / 'models' / 'Meta-Llama-3-8B-Instruct.Q5_K_M.gguf'
PREPROCESSOR_DIR: Path = ROOT_DIR / 'training' / 'models'/ 'preprocesssor.b'
BEST_MODELS_DIR: Path = ROOT_DIR / 'training' / 'models' 

# COLUMNS OF THE DATASET
COLUMNS_TO_DROP_KFOLD: List[str] = ['id', 'label', 'event_timestamp', 'kfold']
COLUMNS_TO_DROP_TUNING: List[str] = ['id', 'label', 'event_timestamp']
TARGET_COLUMN: str = 'label'
KFOLD_COLUMN: str = 'kfold'

# SKLEARN 
SKLEARN_MODELS_RANDOM_STATE: int = 197382465
NUMBER_FOLDS: int = 2
TRAIN_TEST_SPLIT_SIZE: float = 0.3

# HYPERPARAMETER TUNING
NUMBER_OF_JOBS: int = -1
HP_TUNING_N_ITER = 5
HP_TUNING_SCORING: List[str] = ['precision', 'recall', 'f1', 'accuracy','roc_auc']
HYPEROPT_MAX_EVALS :int = 2

# MLFLOW 
MLFLOW_TRACKING_URI: str = "http://localhost:5001"

# CONFIG VALIDATION
EXPERIMENT_CONFIG_REQUIRED_KEYS: List[str] = ['experiment_name', 'train_with_tuning', 'model_name', 'preprocessor_method', 'tuning_method', 'model_params']

# BERT
MODEL_CHECK_POINT = 'distilbert-base-uncased'
MODEL_MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
DEVICE = 'cuda'
LEARNING_RATE = 1e-5
NUMBER_EPOCHS: int = 2


class ExplicitEnum(str, Enum):
    def __str__(self):
        return self.value
    
    @classmethod
    def list(cls):
        return [member.value for member in cls.__members__.values()]
    

class Models(ExplicitEnum):
    LOGISTIC_REGRESSION = 'logistic_regression'
    RANDOM_FOREST = 'random_forest'
    XGBOOST = 'xgboost'
    LLAMA = 'llama'
    DISTILBERT = 'distilbert'
    

class Preprocessors(ExplicitEnum):
    STANDARD_SCALER = 'standard_scaler'

        
class HyperparameterTuningMethods(ExplicitEnum):
    RANDOM_SEARCH = 'random_search'
    HYPEROPT = 'hyperopt'
    
