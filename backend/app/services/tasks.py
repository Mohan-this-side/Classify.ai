"""
Celery tasks to run workflows asynchronously.
"""

from .celery_app import celery_app
from ..workflows.classification_workflow import ClassificationWorkflow


@celery_app.task(name="run_workflow")
def run_workflow_task(serialized_df: str, target_column: str, description: str, user_id: str) -> dict:
    # In a real setup, we'd load dataset from object storage; here we keep Celery scaffold minimal.
    # This task is provided for future integration and not used by the current FastAPI background tasks.
    workflow = ClassificationWorkflow()
    # Placeholder: skip because we don't deserialize DataFrame here.
    return {"status": "not_implemented"}


