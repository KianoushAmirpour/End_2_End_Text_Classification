import re
import nltk
import string
import pathlib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import List, Pattern


class ColumnsConfig:
    COLUMNS_NEW_NAMES: List[str] = ['qid', 'post', 'label']
    IGNORED_COLUMNS_FOR_QUALITY_CHECK: List[str] = ['id', 'label', 'event_timestamp']
    COLUMN_TO_BE_CLEANED: str = 'post'
    COLUMN_FOR_META_FEATURES: str = 'post'
    CLEANED_COLUMN: str = 'cleaned'
    STEMMED_COLUMN: str = 'stemmed'
    EMBEDDINGS_COLUMN: str = 'embeddings'


class DirectoriesConfig:
    ROOT_DIR: pathlib.Path = pathlib.Path(__file__).parents[2]
    FEATURE_STORE_DIR: pathlib.Path = ROOT_DIR / 'feature_store'/'feature_repo'
    FEATURE_PIPELINE_LOGGING_PATH: pathlib.Path = ROOT_DIR / 'feature_pipeline' / 'custom_logs'/ 'feature_pipeline.log'
    TEMP_DIR: pathlib.Path = ROOT_DIR / 'feature_pipeline' / 'temp'


class TextCleaningConfig:
    try:
        ENG_STOP_WORDS: List[str] = stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
        ENG_STOP_WORDS: List[str] = stopwords.words('english')
   
    # punctuations
    PUNCTUATIONS: str = string.punctuation

    # regex pattern for text cleaning
    HTML_TAGS_PATTERN: Pattern[str] = re.compile('<.*?>')
    URL_PATTERN: Pattern[str] = re.compile('https?://[^\s]+|www\.[^\s]+')
    EMOJE_PATTERN: Pattern[str] = re.compile('[\U0001F600-\U0001F64F\u2600-\u26FF\u2700-\u27BF\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF]+')
    SPECIAL_CHARS_PATTERN: Pattern[str] = re.compile('[^\w\s]')

    # roots of words
    STEMMER: PorterStemmer = PorterStemmer()
    DEFAULT_TECHNIQUE: str = 'stem'


class EmbeddingConfig:
    model_name : str = 'all-MiniLM-L6-v2'
