import logging
from http import HTTPStatus

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response

from ai_service.api.auth import require_api_key
from ai_service.config import Settings
from ai_service.errors import (
    InvalidModelOutput,
    InvalidTaskInput,
    OllamaError,
    OllamaTimeout,
    OllamaUnavailable,
)
from ai_service.models.api import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
)
from ai_service.ollama import OllamaClient
from ai_service.services.generation import GenerationService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    service: GenerationService = http_request.app.state.generation_service
    chat_result = await service.chat(request)
    settings: Settings = http_request.app.state.settings
    logger.info("Completed chat request using model %s", settings.model_name)
    return ChatResponse(
        response=chat_result["response"],
        model=settings.model_name,
        metrics=chat_result["metrics"],
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    service: GenerationService = http_request.app.state.generation_service
    try:
        generation_result = await service.generate(request)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown task: {request.task}",
        ) from exc
    except InvalidTaskInput as exc:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    settings: Settings = http_request.app.state.settings
    logger.info("Completed generation task %s", request.task)
    return GenerateResponse(
        task=request.task,
        result=generation_result["result"],
        model=settings.model_name,
        metrics=generation_result["metrics"],
    )


@router.get("/health", response_model=HealthResponse)
async def health(http_request: Request, response: Response) -> HealthResponse:
    ollama: OllamaClient = http_request.app.state.ollama_client
    settings: Settings = http_request.app.state.settings
    try:
        models = await ollama.list_models()
    except OllamaError as exc:
        logger.warning("Health check could not reach Ollama: %s", exc)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            ollama="unavailable",
            configured_model=settings.model_name,
            model_available=False,
            error=str(exc),
        )

    model_available = any(
        model.get("name") == settings.model_name
        for model in models
        if isinstance(model.get("name"), str)
    )
    response.status_code = (
        status.HTTP_200_OK if model_available else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return HealthResponse(
        status="ok" if model_available else "degraded",
        ollama="available",
        configured_model=settings.model_name,
        model_available=model_available,
    )


@router.get("/models", response_model=list[ModelInfo])
async def models(http_request: Request) -> list[ModelInfo]:
    ollama: OllamaClient = http_request.app.state.ollama_client
    available = await ollama.list_models()
    return [
        ModelInfo(
            name=model["name"],
            size=model.get("size"),
            modified_at=model.get("modified_at"),
        )
        for model in available
        if isinstance(model.get("name"), str)
    ]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(OllamaError)
    async def handle_ollama_error(_: Request, exc: OllamaError) -> JSONResponse:
        logger.warning("Ollama request failed: %s", exc)
        if isinstance(exc, OllamaTimeout):
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
        elif isinstance(exc, OllamaUnavailable):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            status_code = status.HTTP_502_BAD_GATEWAY
        return JSONResponse(
            status_code=status_code,
            content={"detail": str(exc)},
        )

    @app.exception_handler(InvalidModelOutput)
    async def handle_invalid_model_output(
        _: Request, exc: InvalidModelOutput
    ) -> JSONResponse:
        logger.warning("Invalid structured model output: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(exc)},
        )
