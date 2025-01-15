from typing import List

import pandas as pd  # type: ignore

from xgboost_model.constants import (
    ENG_STOP_WORDS,
    PUNCTUATIONS
    )  # type: ignore


def count_stop_words(words: List[str]) -> int:
    return sum(
        word.lower() in ENG_STOP_WORDS for word in words)


def count_title_case(words: List[str]) -> int:
    return sum(word.istitle() for word in words)


def cal_ave_length(words: List[str]) -> float:
    return sum([len(word) for word in words]) / len(words)


def count_num_punctuations(text: str) -> int:
    return sum(char in PUNCTUATIONS for char in text)


def generate_features(input_data: pd.DataFrame) -> pd.DataFrame:
    df = input_data.copy()
    df['words'] = df['post'].str.split()
    df['num_words'] = df['words'].apply(len)
    df['num_unique_words'] = df['words'].apply(lambda x: len(set(x)))
    df['num_stop_words'] = df['words'].apply(lambda x: count_stop_words(x))
    df['num_title_case'] = df['words'].apply(lambda x: count_title_case(x))
    df['ave_length_words'] = df['words'].apply(lambda x: cal_ave_length(x))
    df['num_characters'] = df['post'].str.len()
    df.drop(columns=['post', "words"], inplace=True)
    return df
