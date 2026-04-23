from .app import app, fastapi_app, run_compare, setup
from .models import (
    CompareRequest,
    CompareResponse,
    StatusResponse,
)

__all__ = [
    "app",
    "run_compare",
    "setup",
    "fastapi_app",
    "CompareRequest",
    "CompareResponse",
    "StatusResponse",
]

# modal deploy -m modal_app --env cytetrainer
