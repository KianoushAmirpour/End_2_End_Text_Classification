import pandas as pd
from .logger import setup_logger
from .config import DirectoriesConfig, ColumnsConfig
from .data_quality_check import (
    check_required_columns,
    check_minimum_num_rows,
    check_null_values
    )

logger = setup_logger(__name__)


def preprocess() -> None:
    
    logger.info('Starting data preprocessing...')
    
    try:
        df = pd.read_csv(DirectoriesConfig.FEATURE_STORE_DIR / 'data'/ 'train.csv')
        logger.info('Csv file is loaded as pandas dataframe successfully.')
    except Exception as e:
        logger.error(f'An error occured while loading the csv file. The error was: {e}.')
        raise
    
    # renaming columns
    if len(ColumnsConfig.COLUMNS_NEW_NAMES) == len(df.columns.to_list()):
        try:
            df.columns = ColumnsConfig.COLUMNS_NEW_NAMES
            # df.rename(columns=ColumnsConfig.COLUMNS_NEW_NAMES, inplace=True)
            logger.info('Columns renamed successfully.')
        except Exception as e:
            logger.error(
                f'An error occured while renaming the columns. The error was: {e}.'
            )
            raise
    else:
        logger.error('The number of columns in the new names list is not equal to the number of columns in the dataframe.')
        raise
    
    try:
        df['id'] = list(range(1, len(df) + 1))
        logger.info('Column `id` added successfully.')
    except Exception as e:
        logger.error(f'An error occured while adding column `id`. The error was: {e}.')
        raise
    
    try:
        timestamps = pd.date_range(end=pd.Timestamp.now(tz='UTC'),
                                    periods=len(df),
                                    freq='s').to_frame(name='event_timestamp', index=False)

        df = pd.concat([df, timestamps], axis=1)
        logger.info('Column `event_timestamp` added successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while adding column `event_timestamp`. The error was: {e}.')
        raise
    
    try:
        df.drop(columns=['qid'], inplace=True)
        logger.info('Column `qid` dropped successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while dropping the column `qid`. The error was: {e}.'
        )
        raise
    
    try:
        df = df[~(df['post'].isnull() | df['label'].isnull())]
        logger.info('Null values removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the null values. The error was: {e}.'
        )
        raise
    
    try:
        df = df[df['label'].apply(lambda x: isinstance(x, int))]
        logger.info('Non integer values removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the non integer values. The error was: {e}.'
        )
        raise        
    
    try:
        df.drop_duplicates(subset=['post'], inplace=True)
        logger.info('Duplicate rows removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the duplicate rows. The error was: {e}.'
        )
        raise
    
    try:
        df = df.loc[:, ['id', 'event_timestamp', 'post', 'label']]
        logger.info('columns reordered successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while reordering the columns. The error was: {e}.'
        )
        raise
    
    # data quality checks
    check_required_columns(df, ['id', 'event_timestamp', 'post', 'label'])
    check_minimum_num_rows(df, 1200000)
    check_null_values(df, ['post', 'label'])
    
    try:
        target_df = df[['id', 'event_timestamp', 'label']]
        target_df.to_parquet(DirectoriesConfig.FEATURE_STORE_DIR /
                      'data' / 'target_df.parquet', index=False)
        logger.info('Target dataframe saved successfully as parquet file.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR}. The error was: {e}.')
        raise
    
    try:
        df.to_csv(DirectoriesConfig.TEMP_DIR / 'preprocessed.csv', index=False)
        logger.info(f'Dataframe was saved as csv file at {DirectoriesConfig.TEMP_DIR} successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as csv file at {DirectoriesConfig.TEMP_DIR}. The error was: {e}.'
        )
        raise
    
    try:
        df.to_parquet(DirectoriesConfig.FEATURE_STORE_DIR / 'data' / 'train.parquet', index=False)
        logger.info(f'Dataframe was saved as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR} successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR}. The error was: {e}.'
        )
        raise
    
    logger.info('Data preprocessing completed successfully.')
        
