from ai_service.models.tasks import OrderEvaluation, OrderInput
from ai_service.tasks.registry import TaskDefinition, register_task


register_task(
    TaskDefinition(
        task_id="freelance_order_evaluation",
        input_model=OrderInput,
        output_model=OrderEvaluation,
        prompt_file="order_evaluation.txt",
    )
)
