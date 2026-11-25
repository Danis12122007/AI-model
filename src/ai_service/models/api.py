from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=20_000)
    system_prompt: str | None = Field(default=None, max_length=10_000)
    temperature: float | None = Field(default=None, ge=0, le=2)


class ChatResponse(BaseModel):
    response: str
    model: str
    metrics: dict[str, Any]


class GenerationParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0, le=2)
    num_predict: int | None = Field(default=None, ge=1, le=8192)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=80)
    data: dict[str, Any]
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class GenerateResponse(BaseModel):
    task: str
    result: dict[str, Any]
    model: str
    metrics: dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    size: int | None = None
    modified_at: str | None = None


class HealthResponse(BaseModel):
    status: str
    ollama: str
    configured_model: str
    model_available: bool
    error: str | None = None
