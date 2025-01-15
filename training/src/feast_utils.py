import pandas as pd
from typing import List
from pathlib import Path
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from .logger import setup_logger
from .constants import FEATURE_STORE_DIR

logger = setup_logger(__name__)

class FeatureRetriever:
    def __init__(self, feature_repo_path:Path = FEATURE_STORE_DIR,
                 entity_df_path:Path = FEATURE_STORE_DIR /'data' / 'target_df.parquet') -> None:
        self.feature_store : FeatureStore = FeatureStore(repo_path=feature_repo_path)
        logger.info('Feature store loaded successfully.')
        self.entity_df: pd.DataFrame = pd.read_parquet(entity_df_path)
        logger.info('Entity dataframe loaded successfully.')
        
    def retreive_features(self, features: List[str]) -> pd.DataFrame:
        try:
            df: pd.DataFrame = self.feature_store.get_historical_features(
            entity_df=self.entity_df,
            features=features
        ).to_df()
            logger.info(f'Features retrieved successfully. Dataframe shape: {df.shape},\
                        Dataframe columns: {df.columns.to_list()}')
            return df
        except Exception as e:
            logger.error(f'An error occured while retrieving features. The error was: {e}.')
        
    def create_saved_dataset(self, training_data: pd.DataFrame,
                            name: str, path: Path) -> None:
        try:
            self.feature_store.create_saved_dataset(
                from_=training_data,
                name=name,
                storage=SavedDatasetFileStorage(path)
            )
            logger.info(f'Saved dataset created successfully at {path}')
        except Exception as e:
            logger.error(f'An error occured while saving the dataset. The error was: {e}.')