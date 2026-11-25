import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

from pydantic import ValidationError

from ai_service.config import Settings
from ai_service.errors import InvalidModelOutput, InvalidTaskInput, OllamaTimeout
from ai_service.models.api import ChatRequest, GenerateRequest
from ai_service.ollama import OllamaClient
from ai_service.tasks.registry import get_task

T = TypeVar("T")


class GenerationService:
    def __init__(self, ollama: OllamaClient, settings: Settings) -> None:
        self._ollama = ollama
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    @staticmethod
    def _metrics(model_result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: model_result.get(key)
            for key in (
                "done",
                "done_reason",
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "eval_duration",
            )
        }

    @staticmethod
    def _print_model_result(model_result: dict[str, Any]) -> None:
        print(
            f"done: {model_result.get('done')}, "
            f"done_reason: {model_result.get('done_reason')}, "
            f"prompt_eval_count: {model_result.get('prompt_eval_count')}, "
            f"eval_count: {model_result.get('eval_count')}, "
            f"total_duration: {model_result.get('total_duration')}, "
            f"load_duration: {model_result.get('load_duration')}, "
            f"eval_duration: {model_result.get('eval_duration')}",
            flush=True,
        )

    async def chat(self, request: ChatRequest) -> dict[str, Any]:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        model_result = await self._run(
            self._ollama.chat(
                messages,
                temperature=(
                    request.temperature
                    if request.temperature is not None
                    else self._settings.model_temperature
                ),
            )
        )
        self._print_model_result(model_result)
        return {
            "response": model_result["content"],
            "metrics": self._metrics(model_result),
        }

    async def generate(self, request: GenerateRequest) -> dict[str, Any]:
        definition = get_task(request.task)
        if definition is None:
            raise KeyError(request.task)

        try:
            task_input = definition.input_model.model_validate(request.data)
        except ValidationError as exc:
            raise InvalidTaskInput(str(exc)) from exc

        messages = [
            {"role": "system", "content": definition.load_prompt()},
            {"role": "user", "content": task_input.model_dump_json(exclude_none=True)},
        ]
        temperature = (
            request.parameters.temperature
            if request.parameters.temperature is not None
            else self._settings.model_temperature
        )
        output_schema = definition.output_model.model_json_schema()
        model_result = await self._run(
            self._ollama.chat(
                messages,
                temperature=temperature,
                output_schema=output_schema,
                num_predict=request.parameters.num_predict,
            )
        )
        self._print_model_result(model_result)
        content = model_result["content"]

        try:
            output = definition.output_model.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            raise InvalidModelOutput(
                f"Model output did not match the '{request.task}' response schema"
            ) from exc
        return {
            "result": output.model_dump(mode="json"),
            "metrics": self._metrics(model_result),
        }

    async def _run(self, operation: Awaitable[T]) -> T:
        async def run_with_limit() -> T:
            async with self._semaphore:
                return await operation

        try:
            return await asyncio.wait_for(
                run_with_limit(),
                timeout=self._settings.model_request_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise OllamaTimeout("Model request exceeded the configured timeout") from exc
