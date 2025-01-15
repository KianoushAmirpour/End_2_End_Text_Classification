import pandas as pd
from .logger import setup_logger
from typing import List


logger = setup_logger(__name__)


class DataQualityChecksException(Exception):
    pass


def check_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns.to_list()]
    if missing_columns:
        raise DataQualityChecksException(f"Missing required columns: {', '.join(missing_columns)}.")
    else:
        logger.info('All required columns are present.')


def check_minimum_num_rows(df: pd.DataFrame, min_rows: int) -> None:
    if df.shape[0] < min_rows:
        raise DataQualityChecksException(f"Dataset has less than {min_rows} rows.")
    else:
        logger.info(f"Dataset has {df.shape[0]} rows.")


def check_minimum_value(df: pd.DataFrame, columns: List[str], min_value: float = 0.0) -> None:
    cols_with_negative_vals = [column for column in columns if df[column].min() < min_value]
    if cols_with_negative_vals:
        raise DataQualityChecksException(f"Column {', '.join(cols_with_negative_vals)} has values less than {min_value}.")
    else:
        logger.info(f"Column {', '.join(cols_with_negative_vals)} has values greater than {min_value}.")


def check_null_values(df: pd.DataFrame, columns: List[str]) -> None:
    number_of_null_values = df.isnull().sum().sum()
    
    df_with_empty_string = df[df[columns].eq('').any(axis=1)]
    
    if number_of_null_values > 0 or df_with_empty_string.shape[0] > 0:
        raise DataQualityChecksException("Null values found.")
    else:
        logger.info("No null values found.")
