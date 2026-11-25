from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    prompt_file: str

    def load_prompt(self) -> str:
        return (PROMPTS_DIR / self.prompt_file).read_text(encoding="utf-8")


TASKS: dict[str, TaskDefinition] = {}


def register_task(definition: TaskDefinition) -> None:
    if definition.task_id in TASKS:
        raise ValueError(f"Task is already registered: {definition.task_id}")
    TASKS[definition.task_id] = definition


def get_task(task_id: str) -> TaskDefinition | None:
    return TASKS.get(task_id)
