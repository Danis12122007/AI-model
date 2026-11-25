from typing import Any

import httpx

from ai_service.config import Settings
from ai_service.errors import OllamaResponseError, OllamaTimeout, OllamaUnavailable


class OllamaClient:
    def __init__(self, http_client: httpx.AsyncClient, settings: Settings) -> None:
        self._http = http_client
        self._settings = settings

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            response = await self._http.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Timed out while listing Ollama models") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama returned HTTP {exc.response.status_code} while listing models"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable("Could not connect to Ollama") from exc
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned invalid JSON") from exc

        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise OllamaResponseError("Ollama response did not contain a models list")
        return [model for model in models if isinstance(model, dict)]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        output_schema: dict[str, Any] | None = None,
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._settings.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self._settings.model_keep_alive,
            "options": {
                "temperature": temperature,
                "num_ctx": self._settings.model_context_length,
                "num_predict": (
                    num_predict
                    if num_predict is not None
                    else self._settings.model_max_tokens
                ),
                "num_thread": self._settings.model_num_threads,
            },
        }
        if output_schema is not None:
            payload["format"] = output_schema

        try:
            response = await self._http.post("/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException as exc:
            raise OllamaTimeout("Ollama generation request timed out") from exc
        except httpx.HTTPStatusError as exc:
            raise OllamaResponseError(
                f"Ollama returned HTTP {exc.response.status_code} during generation"
            ) from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable("Could not connect to Ollama") from exc
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned invalid JSON") from exc

        message = result.get("message") if isinstance(result, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise OllamaResponseError("Ollama response did not contain message content")

        return {
            "content": content,
            "done": result.get("done"),
            "done_reason": result.get("done_reason"),
            "prompt_eval_count": result.get("prompt_eval_count"),
            "eval_count": result.get("eval_count"),
            "total_duration": result.get("total_duration"),
            "load_duration": result.get("load_duration"),
            "eval_duration": result.get("eval_duration"),
        }
