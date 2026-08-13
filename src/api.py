from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import predict_spam


app = FastAPI(
    title="Sentinel-NLP Spam Detection API",
    description="Production API for SMS spam classification using a trained NLP model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    message: str = Field(
        min_length=1,
        description="SMS message to classify as spam or ham.",
        examples=[
            "Congratulations! You have won a free prize. Call 08001234567 now."
        ],
    )


class PredictionResponse(BaseModel):
    label: str = Field(
        description="Predicted class: spam or ham.",
        examples=["spam"],
    )
    spam_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Model probability that the message is spam.",
        examples=[0.9671804130816058],
    )
    threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability threshold used to classify the message as spam.",
        examples=[0.3],
    )


@app.get(
    "/health",
    summary="Health check",
    description="Returns the current health status of the API.",
)
def health_check():
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an SMS message",
    description="Classifies an SMS message as spam or ham using the Sentinel-NLP model.",
)
def predict(request: PredictionRequest):
    return predict_spam(request.message)