from .logger import setup_logger
from .config import DirectoriesConfig, ColumnsConfig
import pandas as pd
from .utils import (remove_html_tags,
                    remove_urls,
                    remove_punctuations,
                    remove_emojis,
                    remove_special_chars,
                    remove_stopwords,
                    remove_extra_spaces,
                    get_stemmed_words)
from .data_quality_check import (
    check_required_columns,
    check_minimum_num_rows,
    check_null_values
    )


logger = setup_logger(__name__)


def clean_text() ->None:

    logger.info('Starting generating cleaned text...')

    try:
        df = pd.read_csv(DirectoriesConfig.TEMP_DIR / 'preprocessed.csv')
        logger.info('Csv file is loaded as pandas dataframe successfully.')
    except Exception as e:
        logger.error(
            'An error occured while loading the csv file. The error was: {e}.')
        raise

    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.COLUMN_TO_BE_CLEANED].str.lower()
        logger.info('Text lowercased successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while lowercasing the text. The error was: {e}.')
        raise

    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_html_tags(x))
        logger.info('Html tags removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the html tags. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_urls(x))
        logger.info('Urls removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the urls. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_punctuations(x))
        logger.info('Punctuations removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the punctuations. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_emojis(x))
        logger.info('Emojis removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the emojis. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_special_chars(x))
        logger.info('Special characters removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the special characters. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_stopwords(x))
        logger.info('Stopwords removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the stopwords. The error was: {e}.')
        raise
    
    try:
        df[ColumnsConfig.CLEANED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: remove_extra_spaces(x))
        logger.info('Extra whitespaces removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the extra whitespaces. The error was: {e}.')
        raise
    
    try:
        df = df.dropna(how='any', axis=0)
        df = df[df[ColumnsConfig.CLEANED_COLUMN] != '']
        logger.info('Null values removed successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while removing the empty rows. The error was: {e}.')

    try:
        df[ColumnsConfig.STEMMED_COLUMN] = df[ColumnsConfig.CLEANED_COLUMN].apply(lambda x: get_stemmed_words(x))
        logger.info('Stemmed words generated successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while generating stemmed words. The error was: {e}.')
        raise
    
    try:
        df = df.drop(columns=[ColumnsConfig.COLUMN_TO_BE_CLEANED])
        logger.info('Columns dropped successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while dropping the columns. The error was: {e}.'
        )
        raise
    
    # data quality checks
    check_required_columns(df, [ColumnsConfig.CLEANED_COLUMN, ColumnsConfig.STEMMED_COLUMN, 'label'])
    check_minimum_num_rows(df, 1200000)
    check_null_values(df, [ColumnsConfig.CLEANED_COLUMN, ColumnsConfig.STEMMED_COLUMN, 'label'])
    
    try:
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], utc=True)
        logger.info('Column `event_timestamp` converted to datetime successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while converting the column `event_timestamp` to datetime. The error was: {e}.')
        raise
    
    try:
        df.to_parquet(DirectoriesConfig.FEATURE_STORE_DIR / 'data' / 'cleaned_stemmed.parquet', index=False)
        logger.info('cleaned_stemmed.parquet file saved successfully.')
    except Exception as e:
        logger.error(
            f'An error occured while saving the dataframe as parquet file at {DirectoriesConfig.FEATURE_STORE_DIR}. The error was: {e}.')
        raise
        
    logger.info('Data cleaning completed successfully.')
