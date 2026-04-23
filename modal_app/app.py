"""Modal entry point for the CyteOnto API.

Deploy:
    modal deploy modal_app/app.py  --env cytetrainer

Prime the volume once after deploy:
    modal run modal_app/app.py::setup  --env cytetrainer

Force-refresh the precomputed assets:
    modal run modal_app/app.py::setup --force  --env cytetrainer
"""

import sys
from typing import Any

import modal
from modal.volume import Volume

from .api import create_app
from .config import AppConfig, build_cyteonto_image
from .worker import run_compare_job

app_config = AppConfig()
app = modal.App(app_config.APP_NAME)
image = build_cyteonto_image()
volume = modal.Volume.from_name(
    app_config.VOLUME_NAME, create_if_missing=True, version=2
)
VOLUME_MAP: dict[str, Volume] = {app_config.VOLUME_MOUNT_PATH: volume}
cyteonto_secrets = modal.Secret.from_name(app_config.SECRET_NAME)


@app.function(
    image=image,
    volumes=VOLUME_MAP,
    secrets=[cyteonto_secrets],
    timeout=app_config.MODAL_MAX_TIMEOUT_SECONDS,
    cpu=app_config.WORKER_CPU,
    memory=app_config.WORKER_MEMORY_MB,
)
async def run_compare(run_id: str, payload: dict[str, Any]) -> None:
    """Worker: execute one compare request and update status on the volume."""
    await run_compare_job(run_id, payload, volume)


@app.function(
    image=image,
    volumes=VOLUME_MAP,
    timeout=3600,
    cpu=app_config.WORKER_CPU,
    memory=app_config.WORKER_MEMORY_MB,
)
def setup(force: bool = False) -> int:
    """One-shot hook: download core ontology assets onto the Modal volume."""
    from cyteonto_new.logger import logger
    from cyteonto_new.setup import main as setup_main

    argv = ["setup.py", "--data-dir", app_config.REMOTE_DATA_DIR]
    if force:
        argv.append("--force")

    original = sys.argv[:]
    sys.argv = argv
    try:
        exit_code = setup_main()
    finally:
        sys.argv = original

    if exit_code != 0:
        logger.error(f"setup failed with exit code {exit_code}")
        return exit_code

    volume.commit()
    logger.info(
        f"Volume '{app_config.VOLUME_NAME}' primed under {app_config.REMOTE_DATA_DIR}"
    )
    return 0


@app.function(
    image=image,
    volumes=VOLUME_MAP,
    timeout=900,
    cpu=app_config.API_CPU,
    memory=app_config.API_MEMORY_MB,
)
@modal.asgi_app(custom_domains=app_config.CUSTOM_DOMAINS)
def fastapi_app():
    """HTTP surface. Exposes /compare, /status/{run_id}, /result/{run_id}."""
    return create_app(volume, run_compare)
