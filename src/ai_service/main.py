from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI

from ai_service.api.routes import register_error_handlers, router
from ai_service.config import Settings, get_settings
from ai_service.ollama import OllamaClient
from ai_service.services.generation import GenerationService
from ai_service.tasks import greetings, order_evaluation


def create_app(
    settings: Settings | None = None,
    *,
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if (
            app_settings.api_key is None
            or not app_settings.api_key.get_secret_value()
        ):
            raise RuntimeError("API_KEY must be configured before starting the service")

        async with httpx.AsyncClient(
            base_url=app_settings.ollama_base_url,
            timeout=app_settings.model_request_timeout,
            transport=http_transport,
        ) as http_client:
            ollama = OllamaClient(http_client, app_settings)
            app.state.ollama_client = ollama
            app.state.generation_service = GenerationService(ollama, app_settings)
            yield

    app = FastAPI(
        title="Local AI Service",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.include_router(router)
    register_error_handlers(app)
    return app


app = create_app()
