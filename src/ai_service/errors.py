class OllamaError(Exception):
    """Base class for errors communicating with the Ollama service."""


class OllamaUnavailable(OllamaError):
    """Ollama could not be reached."""


class OllamaTimeout(OllamaError):
    """An Ollama request exceeded its configured timeout."""


class OllamaResponseError(OllamaError):
    """Ollama returned an error or an invalid response."""


class InvalidModelOutput(Exception):
    """The model output did not match the task's required response schema."""


class InvalidTaskInput(Exception):
    """The task input did not match the registered task's input schema."""
