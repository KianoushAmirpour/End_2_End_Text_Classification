import pandas as pd
from .logger import setup_logger
from .config import DirectoriesConfig, ColumnsConfig
from .utils import (
    count_stop_words,
    count_title_case,
    cal_ave_length,
    count_num_punctuations)
from .data_quality_check import (
    check_required_columns,
    check_minimum_num_rows,
    check_minimum_value,
    check_null_values)

logger = setup_logger(__name__)


def calculate_meta_features() -> None:

    logger.info('Starting meta features generation...')

    try:
        df = pd.read_csv(DirectoriesConfig.TEMP_DIR / 'preprocessed.csv')
        logger.info('Csv file is loaded as pandas dataframe successfully.')
    except Exception as e:
        logger.error(
            'An error occured while loading the csv file. The error was: {e}.')
        raise

    if ColumnsConfig.COLUMN_FOR_META_FEATURES not in df.columns.to_list():
        logger.error(
            f'The column `{ColumnsConfig.COLUMN_FOR_META_FEATURES}` does not exist in the dataframe.')
        raise

    try:
        df[ColumnsConfig.COLUMN_FOR_META_FEATURES] = df[ColumnsConfig.COLUMN_FOR_META_FEATURES].astype(
            str)
        logger.info(
            f'Column `{ColumnsConfig.COLUMN_FOR_META_FEATURES}` converted to string successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while converting the column `{ColumnsConfig.COLUMN_FOR_META_FEATURES}` to string. The error was: {e}.')
        raise

    try:
        df['words'] = df[ColumnsConfig.COLUMN_FOR_META_FEATURES].str.split()
        logger.info(
            f'Column `{ColumnsConfig.COLUMN_FOR_META_FEATURES}` split into words successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while splitting the column `{ColumnsConfig.COLUMN_FOR_META_FEATURES}` into words. The error was: {e}.')
        raise

    try:
        df['num_words'] = df['words'].apply(len)
        logger.info('Column `num_words` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_words`. The error was: {e}.')
        raise

    try:
        df['num_unique_words'] = df['words'].apply(lambda x: len(set(x)))
        logger.info('Column `num_unique_words` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_unique_words`. The error was: {e}.')
        raise

    try:
        df['num_stop_words'] = df['words'].apply(lambda x: count_stop_words(x))
        logger.info('Column `num_stop_words` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_stop_words`. The error was: {e}.')
        raise

    try:
        df['num_title_case'] = df['words'].apply(lambda x: count_title_case(x))
        logger.info('Column `num_title_case` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_title_case`. The error was: {e}.')
        raise

    try:
        df['ave_length_words'] = df['words'].apply(lambda x: cal_ave_length(x))
        logger.info('Column `ave_length_words` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `ave_lenght_words`. The error was: {e}.')
        raise

    try:
        df['num_characters'] = df[ColumnsConfig.COLUMN_FOR_META_FEATURES].str.len()
        logger.info('Column `num_characters` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_characters`. The error was: {e}.')
        raise

    try:
        df['num_punctuations'] = df[ColumnsConfig.COLUMN_FOR_META_FEATURES].apply(
            lambda x: count_num_punctuations(x))
        logger.info('Column `num_punctuations` generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating column `num_punctuations`. The error was: {e}.')
        raise

    try:
        df.drop(
            columns=[ColumnsConfig.COLUMN_FOR_META_FEATURES, 'words'], inplace=True)
        logger.info('Columns dropped successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while dropping the columns. The error was: {e}.')
        raise

    try:
        df = df.dropna(how='any', axis=0)
        logger.info('Null values removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the empty rows. The error was: {e}.')

    # data quality check
    check_required_columns(df, ['id', 'event_timestamp', 'label',
                                'num_words', 'num_unique_words', 
                                'num_stop_words','num_title_case',
                                'ave_length_words', 'num_characters',
                                'num_punctuations'])
    
    check_minimum_num_rows(df, 1200000)
    
    check_minimum_value(df, ['label', 'num_words', 'num_unique_words',
                             'num_stop_words', 'num_title_case', 'ave_length_words',
                             'num_characters', 'num_punctuations'])
    
    check_null_values(df, ['num_words', 'num_unique_words', 'num_stop_words',
                           'num_title_case', 'ave_length_words', 'num_characters',
                           'num_punctuations', 'label'])

    try:
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)
        logger.info('Column `event_timestamp` converted to datetime successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while converting the column `event_timestamp` to datetime. The error was: {e}.')
        raise
    
    try:
        df.to_parquet(DirectoriesConfig.FEATURE_STORE_DIR /
                      'data' / 'meta_features.parquet', index=False)
        logger.info('Meta features saved successfully as parquet file.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR}. The error was: {e}.')
        raise

    logger.info('Meta features generation completed successfully.')
