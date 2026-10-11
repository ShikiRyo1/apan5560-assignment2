"""Validated request and response contracts shown in the API documentation."""

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    word: str = Field(description="Input word with surrounding whitespace removed")
    normalized_word: str = Field(description="Lowercase word used for the lookup")
    model: str
    model_version: str
    dimensions: int
    vector: list[float] = Field(description="The complete, unnormalized word vector")


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, max_length=100, examples=["the"])
    length: int = Field(ge=1, le=50, examples=[5])


class TextGenerationResponse(BaseModel):
    generated_text: str


class HealthResponse(BaseModel):
    status: str
    model: str
    model_version: str
    dimensions: int


class ClassificationResponse(BaseModel):
    class_id: int = Field(ge=0, le=9)
    class_name: str
    confidence: float = Field(ge=0, le=1, description="Largest softmax score; not calibrated certainty")
    scores: dict[str, float]
    original_size: list[int] = Field(description="Original width and height")
    input_shape: list[int] = Field(description="NCHW shape after preprocessing")
    model: str
