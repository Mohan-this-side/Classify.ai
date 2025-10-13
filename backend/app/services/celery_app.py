"""
Celery application initialization.

This module exposes a Celery app configured from Settings.
"""

from celery import Celery
from ..config import get_settings


def create_celery_app() -> Celery:
    settings = get_settings()
    app = Celery(
        "ds_capstone",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["app.services.tasks"],
    )
    # Reasonable defaults
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = create_celery_app()


