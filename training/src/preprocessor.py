from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    @abstractmethod
    def create_preprocessor_strategy(self):
        pass
    
class StandardScalerPreprocessor(BasePreprocessor):
    def create_preprocessor_strategy(self):
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()

