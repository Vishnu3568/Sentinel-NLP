from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import predict_spam


app = FastAPI(
    title="Sentinel-NLP Spam Detection API",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    message: str = Field(min_length=1)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    return predict_spam(request.message)