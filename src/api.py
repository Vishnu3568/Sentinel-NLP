import logging
from typing import Annotated

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StringConstraints

from src.inference import predict_spam

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentinel-NLP Spam Detection API",
    description="Production API for SMS spam classification using a trained NLP model.",
    version="1.0.0",
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction service temporarily unavailable."
        },
    )


class PredictionRequest(BaseModel):
    message: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ] = Field(
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