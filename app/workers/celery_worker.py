# Uncomment when you need async task processing (large images, batch jobs).
# Requires: pip install celery redis
# Run worker with: celery -A app.workers.celery_worker worker --loglevel=info

# from celery import Celery
#
# celery_app = Celery(
#     "image_processor",
#     broker="redis://localhost:6379/0",
#     backend="redis://localhost:6379/0",
# )
#
# celery_app.conf.update(
#     task_serializer="json",
#     result_serializer="json",
#     accept_content=["json"],
#     task_track_started=True,
#     task_time_limit=300,
#     worker_prefetch_multiplier=1,
#     worker_concurrency=1,   # one GPU task at a time
# )
