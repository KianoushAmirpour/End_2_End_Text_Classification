from fastapi.testclient import TestClient  # type: ignore

from app.config import settings
from app.main import app

client = TestClient(app)


def test_home_page():
    response = client.get("/api/v1/home-page")
    assert response.status_code == 200
    assert response.json() == {"message": settings.MAIN_PAGE_MESSAGE,
                               "version": settings.VERSION}


def test_prediction(sample_input):
    response = client.post("/api/v1/predict", json=sample_input)
    expected_output = {"predictions": [
        {
            "post": "Why my answers not get any upvotes on Quora?",
            "prediction": "sincere"
        },
        {
            "post": "How do you train a pigeon to send messages?",
            "prediction": "sincere",
        },
    ]}
    assert response.status_code == 200
    assert response.json() == expected_output
