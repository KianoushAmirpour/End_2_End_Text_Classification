from fastapi import FastAPI  # type: ignore

from app.config import settings
from app.routes import api_router

app = FastAPI(title=settings.PROJECT_NAME,
              openapi_url=F"{settings.API_V1_STR}/openapi.json")

app.include_router(api_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn  # type: ignore
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)
    