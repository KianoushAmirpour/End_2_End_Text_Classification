import pandas as pd  # type: ignore
from fastapi import APIRouter, HTTPException  # type: ignore
from xgboost_model.predict import make_prediction  # type: ignore

from app import schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/home-page", response_model=schemas.MainPage, status_code=200)
def home_page():
    return {"message": settings.MAIN_PAGE_MESSAGE, "version": settings.VERSION}


@api_router.post("/predict", response_model=schemas.MultipleResponseSchema,
                 status_code=200)
def predict(inputs: schemas.MultipleInputSchema):
    df = pd.DataFrame([post.text for post in inputs.posts], columns=['post'])
    df['id'] = df.index
    result = make_prediction(df)
    errors = result['errors']
    predicted_classes = result['predictions']
    if errors:
        HTTPException(status_code=400, detail=errors)
    predictions = [
        {"post": post.text, "prediction": prediction}
        for post, prediction in zip(inputs.posts, predicted_classes)
        ]
    return {"predictions": predictions}
