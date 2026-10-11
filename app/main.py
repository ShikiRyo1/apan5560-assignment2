"""Add image classification to the existing embedding and text generation API."""

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, File, UploadFile

from app.classifier import ImageClassifier, InvalidImageError, ImageLimitError, MAX_UPLOAD_BYTES

from app.bigram_model import BigramModel
from app.embeddings import (
    MODEL_NAME,
    InvalidWordError,
    UnknownWordError,
    WordEmbeddingService,
)
from app.schemas import (
    ClassificationResponse,
    EmbeddingResponse,
    HealthResponse,
    TextGenerationRequest,
    TextGenerationResponse,
)

CORPUS = [
    "The cat reads books",
    "The dog reads notes",
    "The cat writes notes",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.embeddings = WordEmbeddingService()
    app.state.bigram = BigramModel(CORPUS)
    app.state.classifier = ImageClassifier()
    yield
    del app.state.embeddings
    del app.state.bigram
    del app.state.classifier


app = FastAPI(
    title="Assignment 2: CIFAR-10 Image Classifier",
    description="The specified CNN, with the previous word-vector and text-generation endpoints.",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Status"])
def read_root():
    return {"message": "Assignment 2 API is running", "documentation": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health(request: Request):
    service = request.app.state.embeddings
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_version": service.model_version,
        "dimensions": service.dimensions,
    }


@app.get(
    "/embedding",
    response_model=EmbeddingResponse,
    tags=["Word embeddings"],
    responses={
        404: {"description": "No pretrained vector exists for the word."},
        422: {"description": "A single alphabetic word is required."},
    },
)
def get_embedding(
    request: Request,
    word: Annotated[
        str,
        Query(min_length=1, max_length=100, description="One English word, for example apple"),
    ],
):
    """Return all 300 components of the lowercase word's static embedding."""
    try:
        return request.app.state.embeddings.embed(word)
    except InvalidWordError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnknownWordError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/generate", response_model=TextGenerationResponse, tags=["Text generation"])
def generate_text(request: Request, body: TextGenerationRequest):
    """Sample up to length words from the small course bigram model."""
    try:
        generated = request.app.state.bigram.generate_text(body.start_word, body.length)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"generated_text": generated}


@app.get("/model", tags=["Image classification"])
def model_info(request: Request):
    """Inspect the bundled CNN checkpoint, class order and preprocessing."""
    return request.app.state.classifier.metadata


@app.post(
    "/classify", response_model=ClassificationResponse, tags=["Image classification"],
    responses={413: {"description": "Image byte or pixel limit exceeded."},
               422: {"description": "Missing, corrupt or unsupported image."}},
)
def classify_image(request: Request, file: Annotated[UploadFile, File(description="PNG/JPEG; maximum 5 MiB and 20 million pixels")]):
    """Return one of the ten CIFAR-10 classes for an uploaded image."""
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        return request.app.state.classifier.classify(content)
    except ImageLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        file.file.close()
