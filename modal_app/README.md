# modal_app

Modal deployment of the `cyteonto_new` package as an HTTP service.

## Quickstart

Check out the [example.py](example.py) file for a quickstart. Edit the `CONFIG` block and run:

```bash
uv run python -m modal_app.example
```

Requires:
- `LLM_API_KEY`: Together / OpenRouter / OpenAI key for description generation.
- `EMBEDDING_MODEL_API_KEY`: Embedding provider key (ignored for ollama).

Also check out the [curl calls](#example-curl-calls) section for more examples. Quick example:

```bash
export CYTEONTO_URL="https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run"
export LLM_API_KEY="..."
export EMBEDDING_MODEL_API_KEY="..."
```

To send a request:

```bash
jq -n \
  --arg llm_key "$LLM_API_KEY" \
  --arg emb_key "$EMBEDDING_MODEL_API_KEY" \
  '{
    authorLabels: ["alveolar macrophage", "regulatory T cell", "CD8-positive, alpha-beta T cell"],
    algorithms: {
      algo1: ["lung macrophage", "Treg", "CD8 T cell"],
      algo2: ["alveolar mac", "T regulatory cell", "cytotoxic T cell"]
    },
    llmProvider: "together",
    llmModel: "moonshotai/Kimi-K2.5",
    llmApiKey: $llm_key,
    embeddingProvider: "openrouter",
    embeddingModel: "qwen/qwen3-embedding-8b",
    embeddingApiKey: $emb_key,
    maxDescriptionConcurrency: 10,
    embeddingMaxConcurrent: 10,
    metric: "cosine_kernel"
  }' \
| curl -sS -X POST "$CYTEONTO_URL/compare" \
    -H 'Content-Type: application/json' \
    -d @-
```

This will return a `run-id` and the status will be `queued`. Poll Status:

```bash
curl -sS "$CYTEONTO_URL/status/<run-id>" | jq
```

Fetch the result:

```bash
# JSON
curl -sS "$CYTEONTO_URL/result/run-61f38d4a-e93c-42f8-abfd-b11731d01673?format=json" | jq
# CSV
curl -sS "$CYTEONTO_URL/result/run-61f38d4a-e93c-42f8-abfd-b11731d01673?format=csv" -o "run-61f38d4a-e93c-42f8-abfd-b11731d01673.csv"
```

Detailed [API reference](#api-reference) below.


## What is CyteOnto

`cyteonto_new` compares two sets of cell type annotations against the [Cell Ontology (CL)](https://obofoundry.org/ontology/cl.html). Given parallel label lists from a study author and one or more annotation algorithms, it:

1. Generates a structured description for every label with an LLM.
2. Embeds those descriptions with a configured embedding model.
3. Matches each embedding to the closest CL term.
4. Scores each author/algorithm pair using an ontology-aware similarity metric (default: a Gaussian kernel on the cosine similarity of the CL term embeddings).
5. Returns a tidy DataFrame with one row per `(algorithm, pair_index)`.

All LLM descriptions and embeddings are cached on disk keyed by the text and embedding model names, so a second call with the same models reuses prior work.

## What this service provides

A Modal app named `cyteonto-api` that exposes three HTTP endpoints:

- `POST /compare` submits a compare job and returns a `runId` immediately.
- `GET /status/{runId}` returns the current job state (`queued`, `running`, `completed`, `failed`).
- `GET /result/{runId}` returns the saved DataFrame as CSV or JSON once the job is complete.

Jobs run on a dedicated worker function (`run_compare`) scaled by Modal. Status and results are written to a Modal volume (`cyteonto`) so they survive restarts and are reachable from subsequent API calls.

The deployed URL is printed by `modal deploy`. The current deployment lives at:

```
https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run
```

## Layout

```
modal_app/
├── __init__.py        Public exports (app, functions, request/response models)
├── app.py             modal.App definition, image, volume, three functions
├── api.py             FastAPI factory with /compare, /status, /result
├── config.py          AppConfig, timeouts, image builder, pyproject reader
├── models.py          Pydantic request/response models
└── worker.py          Agent builder and the compare-job body
```

All file-system paths live under `AppConfig` in `config.py`. Runtime constants (CPU, memory, timeouts, model defaults) live on the same class and can be edited without touching the rest of the package.

## Deploy and prime the volume

From the project root:

```bash
uv run modal deploy -m modal_app --env cytetrainer
```

On first deploy (or when you want to refresh the shipped CL files and the precomputed ontology descriptions and embeddings), run the `setup` hook:

```bash
uv run modal run -m modal_app --env cytetrainer setup

uv run modal run -m modal_app --env cytetrainer setup --force
```

`setup` downloads the CL CSV, the OWL file, and the precomputed Kimi-K2.5 descriptions and qwen3-embedding-8b embeddings into the `cyteonto` Modal volume. It is idempotent; `--force` overwrites existing files.

## API reference

Base URL:

```
https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run
```

### POST `/compare`

Submit a new compare job. Returns immediately with a `runId` and initial state `queued`.

Request body:

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `authorLabels` | `list[str]` | yes | | Reference labels, one per item. |
| `algorithms` | `dict[str, list[str]]` | yes | | Map of algorithm name to that algorithm's labels. Each list must have the same length as `authorLabels`. Algorithm names must be unique. |
| `llmProvider` | `openrouter \| together \| openai` | no | `openrouter` | OpenAI-compatible provider used for description generation. |
| `llmModel` | `str` | no | `moonshotai/Kimi-K2.5` | Model name for the selected provider. |
| `llmApiKey` | `str` | yes | | Caller-provided API key for the LLM provider. |
| `embeddingProvider` | `deepinfra \| ollama \| openai \| google \| openrouter \| together` | no | `openrouter` | Embedding backend. |
| `embeddingModel` | `str` | no | `qwen/qwen3-embedding-8b` | Embedding model name. |
| `embeddingApiKey` | `str` | yes | | Caller-provided API key for the embedding provider. Ignored for `ollama`. |
| `embeddingModelSettings` | `dict \| null` | no | `null` | Merged into the embedding request body when `embeddingProvider=openrouter`. Pass `{}` to disable the default DeepInfra routing; pass your own `{"provider": {...}}` to customise. |
| `embeddingMaxConcurrent` | `int >= 1` | no | `100` | Concurrency cap for embedding requests. |
| `maxDescriptionConcurrency` | `int >= 1` | no | `100` | Concurrency cap for LLM description calls. |
| `usePubmedTool` | `bool` | no | `false` | If `true`, the LLM can call a PubMed abstract tool while generating each description. |
| `reasoning` | `bool` | no | `false` | Enable provider reasoning mode. Keep `false` for most use cases. |
| `metric` | `str` | no | `cosine_kernel` | See the `cyteonto_new` README for the full list. |
| `metricParams` | `dict \| null` | no | `null` | Metric-specific parameters (for example `{"width": 0.25}` for `cosine_kernel`). |
| `minMatchSimilarity` | `float in [0, 1]` | no | `0.1` | Threshold below which a label is considered unmatched to any CL term. |
| `useCache` | `bool` | no | `true` | If `false`, all on-disk caches are bypassed for this run. |

Response:

```json
{ "runId": "run-<uuid4>", "state": "queued" }
```

Validation errors return `400` with a JSON `detail` message (for example mismatched label lengths).

### GET `/status/{runId}`

Return the current status record.

Response schema:

| Field | Type | Notes |
|-------|------|-------|
| `runId` | `str` | |
| `state` | `queued \| running \| completed \| failed` | |
| `createdAt` | ISO 8601 `str` | When the request was accepted. |
| `startedAt` | ISO 8601 `str \| null` | When the worker started. |
| `completedAt` | ISO 8601 `str \| null` | When the job reached a terminal state. |
| `error` | `str \| null` | Populated when `state=failed`. `"<ExceptionClass>: <message>"`. |
| `numAuthorLabels` | `int` | |
| `numAlgorithms` | `int` | |
| `numRows` | `int \| null` | Number of rows in the result DataFrame, when `state=completed`. |
| `resultCsvPath` | `str \| null` | Volume-relative path to the CSV result. |
| `resultJsonPath` | `str \| null` | Volume-relative path to the JSON result. |

Returns `404` if the `runId` is unknown.

### GET `/result/{runId}?format=json|csv`

Return the saved DataFrame for a completed run. `format` defaults to `json`.

- `format=json` returns `application/json` with one object per row.
- `format=csv` returns `text/csv` as a downloadable file named `{runId}.csv`.

Returns `404` if the `runId` is unknown and `409` if the run is not in state `completed`.

### Result DataFrame columns

| Column | Type | Meaning |
|--------|------|---------|
| `run_id` | `str` | Echoed from the request. |
| `algorithm` | `str` | Key from the `algorithms` map. |
| `pair_index` | `int` | Position in the label list, starting at 0. |
| `author_label` | `str` | Author label for this pair. |
| `algorithm_label` | `str` | Algorithm label for this pair. |
| `author_ontology_id` | `str \| null` | Best CL match for the author label. |
| `author_embedding_similarity` | `float` | Cosine similarity to that CL term. |
| `algorithm_ontology_id` | `str \| null` | Best CL match for the algorithm label. |
| `algorithm_embedding_similarity` | `float` | Cosine similarity to that CL term. |
| `cytescore_similarity` | `float` | Score under the chosen `metric`; `0.0` if either side is unmatched. |
| `similarity_method` | `str` | `cytescore`, `string_similarity`, `partial_match`, or `no_matches`. |

### GET `/health`

Returns `{"ok": true}`. Useful as a liveness probe.

## Example curl calls

Set the base URL and API keys once:

```bash
export CYTEONTO_URL="https://nygen-labs-cytetrainer--cyteonto-api-fastapi-app.modal.run"
export LLM_API_KEY="..."
export EMBEDDING_MODEL_API_KEY="..."
```

> Shell quoting note: bash does not expand `$VAR` inside single-quoted strings. Do **not** put `"$LLM_API_KEY"` inside `-d '{ ... }'`, the literal string `$LLM_API_KEY` will be sent. Use the `jq` pattern below or switch the body to double quotes with escaped inner quotes.

### Submit a compare job

Recommended pattern, builds the JSON with `jq` so the API keys are interpolated safely:

```bash
jq -n \
  --arg llm_key "$LLM_API_KEY" \
  --arg emb_key "$EMBEDDING_MODEL_API_KEY" \
  '{
    authorLabels: ["alveolar macrophage", "regulatory T cell", "CD8-positive, alpha-beta T cell"],
    algorithms: {
      algo1: ["lung macrophage", "Treg", "CD8 T cell"],
      algo2: ["alveolar mac", "T regulatory cell", "cytotoxic T cell"]
    },
    llmApiKey: $llm_key,
    embeddingApiKey: $emb_key
  }' \
| curl -sS -X POST "$CYTEONTO_URL/compare" \
    -H 'Content-Type: application/json' \
    -d @-
```

Example response:

```json
{ "runId": "run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3", "state": "queued" }
```

Save the id for polling:

```bash
RUN_ID="run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3"
```

### Submit with explicit models and metric parameters

```bash
jq -n \
  --arg llm_key "$LLM_API_KEY" \
  --arg emb_key "$EMBEDDING_MODEL_API_KEY" \
  '{
    authorLabels: ["alveolar macrophage", "regulatory T cell"],
    algorithms: { algo1: ["lung macrophage", "Treg"] },
    llmProvider: "together",
    llmModel: "moonshotai/Kimi-K2.5",
    llmApiKey: $llm_key,
    embeddingProvider: "openrouter",
    embeddingModel: "qwen/qwen3-embedding-8b",
    embeddingApiKey: $emb_key,
    embeddingModelSettings: {
      provider: { order: ["DeepInfra"], allow_fallbacks: true }
    },
    maxDescriptionConcurrency: 50,
    embeddingMaxConcurrent: 50,
    metric: "cosine_kernel",
    metricParams: { center: 1.0, width: 0.25, amplitude: 1.0 },
    minMatchSimilarity: 0.15,
    useCache: true
  }' \
| curl -sS -X POST "$CYTEONTO_URL/compare" \
    -H 'Content-Type: application/json' \
    -d @-
```

### Poll status

```bash
curl -sS "$CYTEONTO_URL/status/$RUN_ID" | jq
```

Example completed status:

```json
{
  "runId": "run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3",
  "state": "completed",
  "createdAt": "2026-04-22T22:11:04.501Z",
  "startedAt": "2026-04-22T22:11:05.812Z",
  "completedAt": "2026-04-22T22:12:48.133Z",
  "error": null,
  "numAuthorLabels": 3,
  "numAlgorithms": 2,
  "numRows": 6,
  "resultCsvPath": "user_files/run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3/result.csv",
  "resultJsonPath": "user_files/run-9b0f1c1e-1c4c-4f3f-a6ad-3fa1e4a5e8c3/result.json"
}
```

Example failed status:

```json
{
  "runId": "run-...",
  "state": "failed",
  "error": "RuntimeError: Failed to embed labels for 'author'",
  "completedAt": "2026-04-22T22:11:34.012Z"
}
```

### Fetch the result

As JSON:

```bash
curl -sS "$CYTEONTO_URL/result/$RUN_ID" | jq
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=json" | jq
```

As CSV:

```bash
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=csv" -o "${RUN_ID}.csv"
```

If the run is still running or has failed, the endpoint returns `409` with a `detail` explaining the current state.

### Health check

```bash
curl -sS "$CYTEONTO_URL/health"
```

## On-disk layout (Modal volume `cyteonto`)

Mounted at `/cyteonto_data` inside every container:

```
/cyteonto_data/
├── cell_ontology/
│   ├── cell_to_cell_ontology.csv
│   └── cl.owl
├── embedding/
│   ├── cell_ontology/
│   │   └── embeddings_<text>_<embd>.npz
│   └── descriptions/
│       └── descriptions_<text>.json
└── user_files/
    └── <run_id>/
        ├── status.json
        ├── result.csv
        ├── result.json
        ├── embeddings/
        │   ├── author/author_embeddings_<text>_<embd>.npz
        │   └── algorithm/<algo_name>_embeddings_<text>_<embd>.npz
        └── descriptions/
            ├── author/author_descriptions_<text>.json
            └── algorithm/<algo_name>_descriptions_<text>.json
```

`<text>` is the cleaned LLM model name (for example `moonshotai-Kimi-K2.5`) and `<embd>` is the cleaned embedding model name. Descriptions and embeddings are cached per text and embedding model, so switching either model produces a fresh cache without invalidating the existing one.

You can inspect the volume directly:

```bash
modal volume ls cyteonto --env cytetrainer
modal volume get cyteonto --env cytetrainer user_files/$RUN_ID/status.json ./status.json
```

## Tuning

All tunables live on `AppConfig` in `modal_app/config.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `PYTHON_VERSION` | `"3.13"` | Python version used by the image. |
| `VOLUME_NAME` | `"cyteonto"` | Modal volume name. |
| `VOLUME_MOUNT_PATH` | `"/cyteonto_data"` | Mount path inside containers. |
| `DEFAULT_LLM_PROVIDER` | `"openrouter"` | Default for requests that omit `llmProvider`. |
| `DEFAULT_LLM_MODEL` | `"moonshotai/Kimi-K2.5"` | Default for requests that omit `llmModel`. |
| `DEFAULT_EMBEDDING_PROVIDER` | `"openrouter"` | Default for requests that omit `embeddingProvider`. |
| `DEFAULT_EMBEDDING_MODEL` | `"qwen/qwen3-embedding-8b"` | Default for requests that omit `embeddingModel`. |
| `WORKER_CPU`, `WORKER_MEMORY_MB` | `1.0`, `2048` | Worker container resources. |
| `API_CPU`, `API_MEMORY_MB` | `0.5`, `1024` | ASGI endpoint container resources. |
| `MODAL_MAX_TIMEOUT_SECONDS` | `86400` | Worker function timeout, set at deploy time. Modal's hard cap is 24 hours. |
| `SECONDS_PER_ALGORITHM` | `1800` | Reserved for future per-call timeout overrides. Currently unused by the running service. |

Redeploy with `uv run modal deploy -m modal_app --env cytetrainer` after changing any of these.

## Error behavior

- Validation errors (empty `authorLabels`, empty `algorithms`, mismatched label lengths) return `400` from `POST /compare`.
- Submitting a job does not validate API keys; an invalid key surfaces later as a failed run via `GET /status/{runId}` with `state="failed"` and `error="ModelHTTPError(status=401): ..."`.
- Missing core files (CSV or OWL) on the volume fail the run with `FileNotFoundError`. Run the `setup` hook to prime the volume.
- Per-label LLM failures do not fail the run. After retries the label gets a blank description that is not persisted, and the next call with the same `runId` retries only those labels.
- A per-request embedding failure (after provider-level retries) aborts the run with `RuntimeError`.

## Related

- `cyteonto_new/README.md` covers the library, the similarity metrics, and the caching rules in depth.
