
import pandas as pd  # type: ignore
from pydantic import BaseModel, ValidationError  # type: ignore


def validate_input(input_data: pd.DataFrame):
    errors = None
    try:
        MultipleInputSchema(inputs=input_data.to_dict(orient='records'))
    except ValidationError as error:
        errors = error.json()
    return input_data, errors


class ModelInputSchema(BaseModel):
    num_words: int
    num_unique_words: int
    num_stop_words: int
    num_title_case: int
    ave_length_words: float
    num_characters: int


class MultipleInputSchema(BaseModel):
    inputs: list[ModelInputSchema]
