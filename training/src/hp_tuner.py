from abc import ABC, abstractmethod


class BaseHyperParameterTuner(ABC):
    @abstractmethod
    def create_tuning_strategy(self,*args, **kwargs):
        pass

class RandomSearchTuner(BaseHyperParameterTuner):
    def create_tuning_strategy(self,*args, **kwargs):
        from sklearn.model_selection import RandomizedSearchCV
        return RandomizedSearchCV(*args, **kwargs)
    
class HyperOptTuner(BaseHyperParameterTuner):
    def create_tuning_strategy(self, *args, **kwargs):
        from hyperopt import fmin
        return fmin(*args, **kwargs)
