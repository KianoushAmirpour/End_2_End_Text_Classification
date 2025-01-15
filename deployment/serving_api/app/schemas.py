from pydantic import BaseModel  # type: ignore


class MainPage(BaseModel):
    message: str
    version: str


class InputSchema(BaseModel):
    text: str


class MultipleInputSchema(BaseModel):
    posts: list[InputSchema]


class ResponseSchema(BaseModel):
    post: str
    prediction: str


class MultipleResponseSchema(BaseModel):
    predictions: list[ResponseSchema]
