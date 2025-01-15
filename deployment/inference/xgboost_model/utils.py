import pickle
from pathlib import Path
from typing import Any, List

import pandas as pd  # type: ignore

from xgboost_model.constants import COLUMNS_TO_RENAME, DATASET_DIR  # type: ignore


def load_artifact(path: Path) -> Any:
    with open(path, "rb") as artifact_file:
        artifact = pickle.load(artifact_file)
    return artifact


def load_dataset(file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(DATASET_DIR / file_name)
    dataframe.columns = COLUMNS_TO_RENAME
    return dataframe


def map_output(predictions: List[int]) -> List[str]:
    map_pred_to_word = {1: 'insincere', 0: 'sincere'}
    predicted_result = [map_pred_to_word[i] for i in predictions]
    return predicted_result
