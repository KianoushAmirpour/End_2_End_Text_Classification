import pytest  # type: ignore

from xgboost_model.constants import DATASET_DIR
from xgboost_model.utils import load_dataset


@pytest.fixture()
def sample_input():
    return load_dataset(DATASET_DIR / 'test.csv')
