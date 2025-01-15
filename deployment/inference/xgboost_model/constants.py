import string
from pathlib import Path
from typing import List

from nltk.corpus import stopwords  # type: ignore

ROOT_DIR: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = ROOT_DIR / 'datasets'
ARTIFACTS_ROOT: Path = ROOT_DIR / 'artifacts'

ENG_STOP_WORDS: List[str] = stopwords.words('english')
PUNCTUATIONS: str = string.punctuation

COLUMNS_TO_RENAME: List[str] = ['id', 'post']
COLUMNS_FOR_MODEL: List[str] = ["id", 'num_words', 'num_unique_words',
                                'num_stop_words', 'num_title_case',
                                'ave_length_words', 'num_characters']
