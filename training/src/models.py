from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import AutoModel
import torch.nn as nn
from .constants import MODEL_CHECK_POINT

class BaseModel(ABC):
    """
    Abstract class for all models
    """
    
    @abstractmethod
    def initialize_model(self, **kwargs):
        raise NotImplementedError(f'Must be implemented inside the subclass.')
    

class LogisticRegressionModel(BaseModel):
    def initialize_model(self, configs: Dict[str, Any]):
        from sklearn.linear_model import LogisticRegression
        from .configs import LogisticRegressionConfig
        model_configs = LogisticRegressionConfig(configs)
        return LogisticRegression(**model_configs.to_dict())

  
class RandomForestModel(BaseModel):
    def initialize_model(self, configs: Dict[str, Any]):
        from sklearn.ensemble import RandomForestClassifier
        from .configs import RandomForestConfig
        model_configs = RandomForestConfig(configs)
        return RandomForestClassifier(**model_configs.to_dict())
    
class XgboostModel(BaseModel):
    def initialize_model(self, configs: Dict[str, Any]):
        from xgboost import XGBClassifier
        from .configs import XGboostConfig
        model_configs = XGboostConfig(configs)
        return XGBClassifier(**model_configs.to_dict(), objective='binary:logistic', tree_method= 'gpu_hist')


class DistilBertModel(nn.Module):
    def __init__(self):
        super(DistilBertModel, self).__init__()
        self.model = AutoModel.from_pretrained(MODEL_CHECK_POINT)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_State = embeddings['last_hidden_state'][:,0]
        output = self.dropout(last_hidden_State)
        output = self.linear(output)
        return output
        