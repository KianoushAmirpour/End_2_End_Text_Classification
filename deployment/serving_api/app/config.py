from pydantic import BaseModel  # type: ignore


class Settings(BaseModel):
    PROJECT_NAME: str = 'Toxic comment classification'
    API_V1_STR: str = "/api/v1"
    MAIN_PAGE_MESSAGE: str = "This is a toxic comment detection API."
    VERSION: str = "1.0.0"


settings = Settings()
