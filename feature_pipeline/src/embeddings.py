# which model to use for embeddings
from sentence_transformers import SentenceTransformer
import pandas as pd
from .logger import setup_logger
from .config import DirectoriesConfig, EmbeddingConfig, ColumnsConfig
from .data_quality_check import (
    check_required_columns,
    check_minimum_num_rows,
    check_null_values
    )


logger = setup_logger(__name__)


def get_embeddings() -> None:
    
    logger.info('Starting Embeddings process...')
    
    model = SentenceTransformer(model_name_or_path=EmbeddingConfig.model_name)
    
    try:
        df = pd.read_parquet(DirectoriesConfig.FEATURE_STORE_DIR / 'data'/ 'cleaned_stemmed.parquet')
        logger.info('Parquet file is loaded as pandas dataframe successfully.')
    except Exception as e:
        logger.error(f'An error occured while loading the parquet file. The error was: {e}.')
        raise
    
    try:
        sentences = df[ColumnsConfig.CLEANED_COLUMN].to_list()
        logger.info('Sentences loaded successfully.')
        embeddings = model.encode(sentences, batch_size=64)
        df[ColumnsConfig.EMBEDDINGS_COLUMN] = list(embeddings)
        logger.info('Embeddings generated successfully.')
    except Exception as e:
        logger.error(f'An error occured while generating embeddings. The error was: {e}.')
        raise
    
    try:
        df.drop(columns=[ColumnsConfig.CLEANED_COLUMN, ColumnsConfig.STEMMED_COLUMN], inplace=True)
        logger.info('Columns dropped successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while dropping the columns. The error was: {e}.')
        raise
    
    # data quality check
    check_required_columns(df, [ColumnsConfig.EMBEDDINGS_COLUMN, 'event_timestamp', 'id', 'label'])
    check_minimum_num_rows(df, 1200000)
    check_null_values(df, [ColumnsConfig.EMBEDDINGS_COLUMN, 'label'])
    
    try:
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)
        logger.info('Column `event_timestamp` converted to datetime successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while converting the column `event_timestamp` to datetime. The error was: {e}.')
        raise
    
    try:
        df.to_parquet(DirectoriesConfig.FEATURE_STORE_DIR / 'data' / 'embeddings.parquet', index=False)
        logger.info('embeddings.parquet file saved successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR}. The error was: {e}.')
        raise
    
    logger.info('Embeddings process completed successfully.')
