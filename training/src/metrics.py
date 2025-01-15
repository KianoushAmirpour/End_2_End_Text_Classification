from sklearn import metrics
from dataclasses import dataclass, field
from .constants import NUMBER_FOLDS
from typing import List, Dict
from .logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

@dataclass
class MetricsCollector:
    train_acccuracy: List[float] = field(default_factory=list)
    validation_accuracy: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    f1_score: List[float] = field(default_factory=list)
    roc_auc: List[float] = field(default_factory=list)
    
    def cal_validation_metrics(self, y_true: np.ndarray, y_pred:np.ndarray) -> None:
        self.validation_accuracy.append(metrics.accuracy_score(y_true, y_pred))
        self.precision.append(metrics.precision_score(y_true, y_pred))
        self.recall.append(metrics.recall_score(y_true, y_pred))
        self.f1_score.append(metrics.f1_score(y_true, y_pred))
            
    def cal_train_metrics(self, y_true:np.ndarray, y_pred:np.ndarray) -> None:
        self.train_acccuracy.append(metrics.accuracy_score(y_true, y_pred))
        
    def cal_roc_auc(self, y_true: np.ndarray, y_prob: np.ndarray)-> None:
        self.roc_auc.append(metrics.roc_auc_score(y_true, y_prob))
        
    def aggregate(self) -> Dict[str, float]:
        agg_metrics = {
            'train_acccuracy_mean': np.mean(self.train_acccuracy),
            'validation_accuracy_mean': np.mean(self.validation_accuracy),
            'precision_mean': np.mean(self.precision),
            'recall_mean': np.mean(self.recall),
            'f1_score_mean': np.mean(self.f1_score),
            'roc_auc_mean': np.mean(self.roc_auc),
            'train_acccuracy_std': np.std(self.train_acccuracy),
            'validation_accuracy_std': np.std(self.validation_accuracy),
            'precision_std': np.std(self.precision),
            'recall_std': np.std(self.recall),
            'f1_score_std': np.std(self.f1_score),
            'roc_auc_std': np.std(self.roc_auc)
        }
        logger.info(f'Metrics aggregated successfully')
        return agg_metrics
    
    @classmethod    
    def fields(cls):
        return cls.__dataclass_fields__
    
    def check_length(self):
        error_message = f"The length of the field '{name}' is {len(getattr(self, name))}, but it must be {NUMBER_FOLDS}. "
        for name, _ in MetricsCollector.fields().items():
            if len(getattr(self, name)) != NUMBER_FOLDS:
                logger.error(error_message)
                raise ValueError(error_message)
        logger.info(f'Length of all calculated metrics for each fold is correct') 
    
