from fastapi import FastAPI
from pydantic import BaseModel
import os

APP_VERSION = os.getenv("APP_VERSION", "0.0.1")

app = FastAPI(title="Agent Factory API", version=APP_VERSION)


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(status="ok", version=APP_VERSION)


@app.get("/version", tags=["Meta"])
def version():
    return {"version": APP_VERSION}
