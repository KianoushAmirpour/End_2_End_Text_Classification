import os
from .logger import setup_logger
from .config import TextCleaningConfig
from typing import List
import re

logger = setup_logger(__name__)


def remove_file(path: str) -> None:
    try:
        os.remove(path)
        logger.info(f'File removed successfully at {path}.')
    except FileNotFoundError:
        logger.error(f'File not found at {path}.')
        raise
    except Exception as e:
        logger.error(
            f'An error occured while removing the file at {path}. The error was: {e}.')
        raise


def remove_stopwords(text: str) -> str:
    return " ".join([word for word in text.split() if word not in set(TextCleaningConfig.ENG_STOP_WORDS)])


def remove_punctuations(text: str) -> str:
    return " ".join([word for word in text.split() if word not in set(TextCleaningConfig.PUNCTUATIONS)])


def remove_html_tags(text: str) -> str:

    return re.sub(TextCleaningConfig.HTML_TAGS_PATTERN, '', text)


def remove_urls(text: str) -> str:

    return re.sub(TextCleaningConfig.URL_PATTERN, "", text)

def remove_extra_spaces(text: str) -> str:

    return " ".join(text.split())


def remove_emojis(text: str) -> str:

    return re.sub(TextCleaningConfig.EMOJE_PATTERN, '', text)

def remove_special_chars(text: str) -> str:

    return re.sub(TextCleaningConfig.SPECIAL_CHARS_PATTERN, ' ', text)


def get_stemmed_words(text: str) -> str:
    return " ".join([TextCleaningConfig.STEMMER.stem(word) for word in text.split()])


def count_stop_words(words: List[str]) -> int:
    return sum(
        word.lower() in TextCleaningConfig.ENG_STOP_WORDS for word in words)


def count_title_case(words: List[str]) -> int:
    return sum(word.istitle() for word in words)


def cal_ave_length(words: List[str]) -> float:
    return sum([len(word) for word in words]) / len(words)


def count_num_punctuations(text: str) -> int:
    return sum(char in TextCleaningConfig.PUNCTUATIONS for char in text)
