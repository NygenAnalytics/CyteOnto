"""Shared configuration for the CyteOnto Modal app."""

from pathlib import Path

import modal
import tomli
from loguru import logger


class AppConfig:
    APP_NAME: str = "cyteonto-api"

    PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent
    PYPROJECT_PATH: Path = PACKAGE_ROOT / "pyproject.toml"
    CYTEONTO_DIR: Path = PACKAGE_ROOT / "cyteonto_new"

    PYTHON_VERSION: str = "3.13"

    VOLUME_NAME: str = "cyteonto"
    VOLUME_MOUNT_PATH: str = "/cyteonto_data"

    REMOTE_DATA_DIR: str = VOLUME_MOUNT_PATH
    REMOTE_USER_DIR: str = f"{VOLUME_MOUNT_PATH}/user_files"
    REMOTE_CELL_ONTOLOGY_DIR: str = f"{VOLUME_MOUNT_PATH}/cell_ontology"
    REMOTE_EMBEDDING_DIR: str = f"{VOLUME_MOUNT_PATH}/embedding"

    DEFAULT_LLM_PROVIDER = "openrouter"
    DEFAULT_LLM_MODEL: str = "moonshotai/Kimi-K2.5"
    DEFAULT_EMBEDDING_PROVIDER: str = "openrouter"
    DEFAULT_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"

    WORKER_CPU: float = 1.0
    WORKER_MEMORY_MB: int = 2048
    API_CPU: float = 0.5
    API_MEMORY_MB: int = 1024

    SECONDS_PER_ALGORITHM: int = 30 * 60
    MODAL_MAX_TIMEOUT_SECONDS: int = 24 * 3600


app_config = AppConfig()


def get_dependencies_from_pyproject(
    group_names: list[str] | None = None,
) -> list[str]:
    """Read dependencies from pyproject.toml, optionally merging named dependency groups."""
    if not app_config.PYPROJECT_PATH.exists():
        logger.error(f"Could not find pyproject.toml at {app_config.PYPROJECT_PATH}")
        raise FileNotFoundError(
            f"pyproject.toml not found at {app_config.PYPROJECT_PATH}"
        )

    with open(app_config.PYPROJECT_PATH, "rb") as f:
        data = tomli.load(f)

    deps: list[str] = list(data.get("project", {}).get("dependencies", []))

    if group_names:
        for group in group_names:
            group_deps: list[str] = data.get("dependency-groups", {}).get(group, [])
            deps.extend(d for d in group_deps if isinstance(d, str))

    logger.debug(f"Resolved {len(deps)} dependencies from groups={group_names}")
    return deps


def build_cyteonto_image() -> modal.Image:
    """Build the container image for workers and the API."""
    deps = get_dependencies_from_pyproject()
    image = (
        modal.Image.debian_slim(python_version=app_config.PYTHON_VERSION)
        .uv_pip_install(*deps)
        .add_local_dir(
            str(app_config.CYTEONTO_DIR),
            "/root/cyteonto_new",
            copy=True,
        )
        .add_local_file(
            str(app_config.PYPROJECT_PATH),
            "/root/pyproject.toml",
            copy=True,
        )
    )
    return image
