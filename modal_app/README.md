# Modal Service

Modal deployment of the `cyteonto` package as an HTTP service, live at `https://cyteonto.nygen.io`.

To use the service, see the [hosted API quickstart](https://docs.nygen.io/CyteOnto/getting-started/quickstart-api/) and the [HTTP API reference](https://docs.nygen.io/CyteOnto/reference/http-api/), which lists every request field, the job status, the result formats and the errors. [example.py](example.py) submits a comparison, polls it and saves the result. Edit its `PAYLOAD`, optionally set `LLM_API_KEY` and `EMBEDDING_API_KEY` to send your own keys, and run:

```bash
uv run python -m modal_app.example
```

This file covers the deployment.

## What this service provides

A Modal app named `cyteonto-api` that exposes these HTTP endpoints:

- `POST /compare` submits a compare job and returns a `runId` immediately.
- `GET /status/{runId}` returns the current job state (`queued`, `running`, `completed`, `failed`).
- `GET /result/{runId}` returns the saved DataFrame as CSV or JSON once the job is complete.
- `GET /health` returns `{"ok": true}` as a liveness probe.

Jobs run on a dedicated worker function (`run_compare`) scaled by Modal. Status and results are written to a Modal volume (`cyteonto`) so they survive restarts and are reachable from subsequent API calls.

The service is served from a custom domain, with the auto-generated Modal URL still available as a fallback:

```
https://cyteonto.nygen.io                                          # custom domain (preferred)
https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run # modal-managed URL
```

Both URLs terminate on the same FastAPI app.

Deployment dashboard (Modal):

```
https://modal.com/apps/nygen-labs/cytetrainer/deployed/cyteonto-api
```

## Layout

```
modal_app/
├── __init__.py        Public exports (app, functions, request/response models)
├── app.py             modal.App definition, image, volume, three functions
├── api.py             FastAPI factory with /compare, /status, /result, /health
├── config.py          AppConfig, timeouts, image builder, pyproject reader
├── example.py         Python client: submit a comparison, poll it and save the result
├── models.py          Pydantic request/response models
└── worker.py          Agent builder and the compare-job body
```

All file-system paths live under `AppConfig` in `config.py`. Most runtime constants (CPU, memory, the worker timeout, model defaults) live on the same class and can be edited without touching the rest of the package.

## Deploy and prime the volume (For Nygen Labs Only)

From the project root:

```bash
uv run modal deploy -m modal_app --env cytetrainer
```

Make sure you have `cyteonto-secrets` secret created in the Modal dashboard. You can create it by going to the Modal dashboard and clicking on the "Secrets" tab.

A healthy deploy prints something like:

```
Created objects.
├── Created mount PythonPackage:modal_app
├── Created mount /home/<you>/.../cyteonto
├── Created mount /home/<you>/.../pyproject.toml
├── Created function run_compare.
├── Created function setup.
├── Created web function fastapi_app => https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run
└── Custom domain for fastapi_app => https://cyteonto.nygen.io
```

On first deploy (or when you want to refresh the shipped CL files and the precomputed ontology descriptions and embeddings), run the `setup` hook:

```bash
uv run modal run --env cytetrainer -m modal_app::setup

uv run modal run --env cytetrainer -m modal_app::setup --force
```

`setup` downloads the CL CSV, the OWL file, the enriched CL CSV (or builds it locally if missing from the CDN), and the precomputed primary-pair descriptions and embeddings into the `cyteonto` Modal volume. Backup-pair artifacts are optional: if those downloads fail, setup continues with the primary pair only. It is idempotent; `--force` overwrites existing files.

## Custom domain

The service is attached to `cyteonto.nygen.io` via Modal's custom domain feature. The domain list is configured on `AppConfig.CUSTOM_DOMAINS` in `modal_app/config.py` and passed to `@modal.asgi_app(custom_domains=...)` in `modal_app/app.py`.

The domain is already registered and verified in this workspace. `https://cyteonto.nygen.io` and the Modal-managed URL serve the same FastAPI app.

## API keys

The worker function is attached to the Modal secret `cyteonto-secrets`. When a request omits `llmApiKey` or `embeddingApiKey`, the worker reads the key of the requested provider from the environment variable that `PROVIDER_API_KEY_ENV` in `cyteonto/config.py` names, such as `NEBIUS_API_KEY`; for embeddings, it then falls back to `EMBEDDING_MODEL_API_KEY`. A key in the request always takes precedence.

| Env var | Used for |
|---------|----------|
| `NEBIUS_API_KEY` | The default LLM and embedding provider, `nebius` |
| `FIREWORKS_API_KEY` | The fallback LLM, Kimi K2.6 on `fireworks` |
| `OPENROUTER_API_KEY` | The fallback embedding model, Qwen3-Embedding-8B on `openrouter` |

A request for another provider without a key needs that provider's variable in the secret; otherwise the job fails with `RuntimeError: llmApiKey missing and no env var for llmProvider=...` (or the embedding equivalent). `POST /compare` cannot reject such requests early: it returns `400` only for providers without a variable in `PROVIDER_API_KEY_ENV`, and every provider of the request schema except `ollama`, which needs no key, has one.

To create the secret, or to replace it with `--force`:

```bash
modal secret create cyteonto-secrets \
  NEBIUS_API_KEY="..." \
  FIREWORKS_API_KEY="..." \
  OPENROUTER_API_KEY="..." \
  --env cytetrainer
```

## On-disk layout (Modal volume `cyteonto`)

Mounted at `/cyteonto_data` inside every container:

```
/cyteonto_data/
├── cell_ontology/
│   ├── cell_to_cell_ontology.csv
│   ├── cell_to_cell_ontology_enriched.csv
│   └── cl.owl
├── embedding/
│   ├── cell_ontology/
│   │   └── embeddings_<llmKey>_<embdKey>.npz
│   └── descriptions/
│       └── descriptions_<llmKey>.json
└── user_files/
    ├── <run_id>/
    │   ├── status.json
    │   ├── result.csv
    │   └── result.json
    ├── embeddings/
    │   └── <run_id>/
    │       ├── author/author_embeddings_<llmKey>_<embdKey>.npz
    │       └── algorithm/<algo_name>_embeddings_<llmKey>_<embdKey>.npz
    ├── descriptions/
    │   └── <run_id>/
    │       ├── author/author_descriptions_<llmKey>.json
    │       └── algorithm/<algo_name>_descriptions_<llmKey>.json
    └── decompositions/
        └── <run_id>/
            └── decompositions_<llmKey>.json
```

`<llmKey>` and `<embdKey>` are the model keys of the LLM and the embedding model, such as `nebius_moonshotai-Kimi-K2.6` and `nebius_Qwen-Qwen3-Embedding-8B` (see [Data files and formats](https://docs.nygen.io/CyteOnto/reference/files/)). Descriptions and embeddings are cached per model, so switching either model produces a fresh cache without invalidating the existing one. The enriched CSV adds `label_normalized` for case-insensitive ontology lookups while preserving original synonym casing for display names.

You can inspect the volume directly:

```bash
modal volume ls cyteonto --env cytetrainer
modal volume get cyteonto --env cytetrainer user_files/$RUN_ID/status.json ./status.json
```

## Tuning

Most tunables live on `AppConfig` in `modal_app/config.py`; the timeouts of the `setup` and `fastapi_app` functions are set in `app.py`, and the fallback models in `cyteonto/config.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `PYTHON_VERSION` | `"3.13"` | Python version used by the image. |
| `VOLUME_NAME` | `"cyteonto"` | Modal volume name. |
| `VOLUME_MOUNT_PATH` | `"/cyteonto_data"` | Mount path inside containers. |
| `DEFAULT_LLM_PROVIDER` | `"nebius"` | Default for requests that omit `llmProvider`. |
| `DEFAULT_LLM_MODEL` | `"moonshotai/Kimi-K2.6"` | Default for requests that omit `llmModel`. |
| `DEFAULT_EMBEDDING_PROVIDER` | `"nebius"` | Default for requests that omit `embeddingProvider`. |
| `DEFAULT_EMBEDDING_MODEL` | `"Qwen/Qwen3-Embedding-8B"` | Default for requests that omit `embeddingModel`. |
| `SECRET_NAME` | `"cyteonto-secrets"` | Modal secret attached to the worker, with the API keys described in [API keys](#api-keys). |
| `CUSTOM_DOMAINS` | `["cyteonto.nygen.io"]` | Domains bound to `fastapi_app` via `@modal.asgi_app(custom_domains=...)`. Must be pre-registered and DNS-verified in the Modal dashboard. |
| `WORKER_CPU`, `WORKER_MEMORY_MB` | `1.0`, `2048` | Worker container resources. |
| `API_CPU`, `API_MEMORY_MB` | `0.5`, `1024` | ASGI endpoint container resources. |
| `MODAL_MAX_TIMEOUT_SECONDS` | `86400` | Worker function timeout, set at deploy time. Modal's hard cap is 24 hours. |
| `SECONDS_PER_ALGORITHM` | `1800` | Reserved for future per-call timeout overrides. Currently unused by the running service. |

Redeploy with `uv run modal deploy -m modal_app --env cytetrainer` after changing any of these.

## Errors

[HTTP API errors](https://docs.nygen.io/CyteOnto/reference/http-api/#errors) lists the responses of the endpoints and the failures of jobs. These points concern the deployment:

- Missing core files (the two CSVs or the OWL) on the volume fail every run with `FileNotFoundError`. Run the `setup` hook to prime the volume.
- A run for a model pair other than the precomputed ones first describes or embeds all 2,902 Cell Ontology terms and writes them to the volume, which takes much longer.
- When an embedding request fails, the run switches to the fallback embedding model and then needs the ontology embeddings of the default LLM with that model, `embeddings_nebius_moonshotai-Kimi-K2.6_openrouter_Qwen-Qwen3-Embedding-8B.npz`. The `setup` hook does not download it, so the first run that fails over embeds the 2,902 ontology descriptions with the fallback model and saves the file on the volume, which delays that run.

## Related

- The [CyteOnto documentation](https://docs.nygen.io/CyteOnto/) covers the library, the similarity metrics and the caching rules in depth.
