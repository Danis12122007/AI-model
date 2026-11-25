from ai_service.models.tasks import GreetingInput, GreetingOutput
from ai_service.tasks.registry import TaskDefinition, register_task


register_task(
    TaskDefinition(
        task_id="greeting",
        input_model=GreetingInput,
        output_model=GreetingOutput,
        prompt_file="greeting.txt",
    )
)
