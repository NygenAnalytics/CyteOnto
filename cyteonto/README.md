# cyteonto_new

Semantic comparison of cell type annotations against the [Cell Ontology (CL)](https://obofoundry.org/ontology/cl.html).

Given two parallel lists of cell type labels (one from the study author, one from an annotation algorithm), the package:

1. Generates a structured description for each label using an LLM.
2. Embeds those descriptions with a configured embedding model.
3. Matches each embedding to the closest CL term via cosine similarity.
4. Scores each author/algorithm pair using an ontology-aware similarity metric (default: kernelised cosine on the CL term embeddings).
5. Returns a tidy `pandas.DataFrame` with per-pair scores.

All LLM descriptions and embeddings are persisted on disk and reused across runs.

---

## Package layout

```
cyteonto_new/
├── __init__.py       Public exports
├── config.py         Environment variables (API keys, log level)
├── logger.py         Loguru configuration
├── paths.py          PathConfig, single source of truth for file locations
├── models.py         CellDescription, EmbdConfig, AgentUsage
├── storage.py        NPZ and JSON read/write (ontology + user variants)
├── embed.py          Async HTTP embedding generation
├── describe.py       LLM description generation with optional PubMed tool
├── ontology.py       OntologyMapping (CSV) and OntologySimilarity (OWL + metrics)
├── cyteonto.py       CyteOnto orchestrator class
└── data/             Shipped and generated data (see "On-disk layout")
```

### Module dependency graph

```
                  config
                    |
                  logger
                    |
  models -- paths --+-- storage
                    |
          embed ----+---- describe
                    |
               ontology
                    |
                cyteonto
                    |
                 __init__
```

Lower layers do not import higher ones. The `CyteOnto` class in `cyteonto.py` is the only module that wires everything together.

---

## Public API

Imported from `cyteonto_new`:

| Symbol              | Kind     | Purpose                                              |
|---------------------|----------|------------------------------------------------------|
| `CyteOnto`          | class    | Main orchestrator                                    |
| `CellDescription`   | model    | Structured LLM output for a single cell type         |
| `EmbdConfig`        | model    | Embedding provider configuration                     |
| `AgentUsage`        | model    | LLM request/token/tool tally                         |
| `RESULT_COLUMNS`    | list     | Column order of the DataFrame returned by `compare`  |

### Quick start

```python
import os
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from cyteonto_new import CyteOnto, EmbdConfig

agent = Agent(
    OpenRouterModel(
        "moonshotai/Kimi-K2.5",  # "moonshotai/Kimi-K2.5"
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
)

embd = EmbdConfig(apiKey=os.environ["EMBEDDING_MODEL_API_KEY"])

cyto = await CyteOnto.from_config(agent=agent, embedding=embd)

df = await cyto.compare(
    author_labels=["animal stem cell", "BFU-E", "neutrophilic granuloblast"],
    algorithms={
        "algo1": ["stem cell", "blast forming unit erythroid", "spermatogonium"],
        "algo2": ["neuronal receptor cell", "stem cell", "ovum"],
    },
    run_id="sample_run_001",         # optional; auto-generated UUID if omitted
    metric="cosine_kernel",
)
print(df)
print("run_id used:", df["run_id"].iloc[0])
```

If you omit `run_id`, the call generates one of the form `run-<uuid4>` and logs it at INFO level. Pick it up from the logs or from `df["run_id"].iloc[0]`. The same id is used on disk under `user_files/embeddings/<run_id>/` and `user_files/descriptions/<run_id>/`.

To remove the cached user artifacts for that run later:

```python
cyto.clear_run("sample_run_001")                        # drop everything for the run
cyto.clear_run("sample_run_001", kind="algorithm")      # drop all algorithm caches only
cyto.clear_run("sample_run_001", kind="algorithm", identifier="algo1")  # one algorithm
```

### Result DataFrame

One row per `(algorithm, pair_index)`:

| Column                            | Type        | Meaning                                                                      |
|-----------------------------------|-------------|------------------------------------------------------------------------------|
| `run_id`                          | str         | The `run_id` passed to `compare`.                                            |
| `algorithm`                       | str         | Algorithm key from the `algorithms` mapping.                                 |
| `pair_index`                      | int         | Position inside the label list, starting at 0.                               |
| `author_label`                    | str         | The author label for this pair.                                              |
| `algorithm_label`                 | str         | The algorithm label for this pair.                                           |
| `author_ontology_id`              | str or None | Best CL match for the author label, or `None` if below threshold.            |
| `author_embedding_similarity`     | float       | Cosine similarity between the author embedding and its CL match.             |
| `algorithm_ontology_id`           | str or None | Best CL match for the algorithm label.                                       |
| `algorithm_embedding_similarity`  | float       | Cosine similarity between the algorithm embedding and its CL match.          |
| `cytescore_similarity`            | float       | Score under the chosen `metric`; `0.0` if either side is unmatched.          |
| `similarity_method`               | str         | `cytescore`, `string_similarity`, `partial_match`, or `no_matches`.          |

---

## Configuration

### Environment variables

Read once on import, via `python-dotenv`:

| Variable                    | Purpose                                                   | Default |
|-----------------------------|-----------------------------------------------------------|---------|
| `EMBEDDING_MODEL_API_KEY`   | Fallback `apiKey` when `EmbdConfig.apiKey` is `None`.     | `""`    |
| `NCBI_API_KEY`              | Optional. Raises PubMed rate limits for the agent tool.   | `""`    |
| `CYTEONTO_LOG_LEVEL`        | Loguru log level (`DEBUG`, `INFO`, `WARNING`, ...).       | `INFO`  |
| `CYTEONTO_LOG_FILE`         | Optional log file path with 10 MB rotation.               | unset   |

The LLM agent keys follow whatever the caller passes to `pydantic_ai`; this package does not manage them.

### `EmbdConfig`

```python
EmbdConfig(
    provider="openrouter",          # deepinfra | ollama | openai | google | openrouter | together
    model="qwen/qwen3-embedding-8b",
    apiKey=None,                    # falls back to EMBEDDING_MODEL_API_KEY
    modelSettings=None,             # openrouter-only pass-through; merged into the request body
    maxConcurrent=100,              # concurrency cap for embedding requests
)
```

Field names are camelCase to stay consistent with the surrounding codebase.

Default behavior when the caller writes `EmbdConfig()` (no args):

- `provider="openrouter"`, `model="qwen/qwen3-embedding-8b"`, `maxConcurrent=100`.
- `modelSettings` is auto-populated to `{"provider": {"order": ["DeepInfra"], "allow_fallbacks": True}}` so OpenRouter tries DeepInfra first and falls back to other providers if it is unavailable.
- To disable the default routing, pass `modelSettings={}`. To customise it, pass your own `{"provider": {...}}` dict.
- `modelSettings` is merged into the embedding request payload only when `provider == "openrouter"`; it is ignored for every other provider.

`CyteOnto.__init__` fails fast with `ValueError` if the resolved `apiKey` is empty and the provider is not `ollama`.

### Recommended LLM

The package uses `moonshotai/Kimi-K2.5` from `Together AI` as the suggested default for description generation. The caller still constructs the `pydantic_ai.Agent` (and thereby chooses the LLM provider), so the constant is a convenience pointer, not a hard-coded dependency. Any pydantic-ai supported model works.

---

## `CyteOnto` class

### Constructor

```python
CyteOnto(
    agent: pydantic_ai.Agent,
    embedding: EmbdConfig,
    data_dir: str | Path | None = None,   # defaults to cyteonto_new/data
    user_dir: str | Path | None = None,   # defaults to <data_dir>/user_files
    max_description_concurrency: int = 100,
    use_pubmed_tool: bool = True,
)
```

`__init__` is cheap: it sets up paths and an `OntologyMapping` (which itself lazy-loads the CSV on first access). It does not read the OWL file, the ontology embeddings, or call any network.

### `from_config` (async factory)

```python
cyto = await CyteOnto.from_config(
    agent=agent,
    embedding=embd,
    data_dir=None,
    user_dir=None,
    force_regenerate=False,
    max_description_concurrency=100,
    use_pubmed_tool=True,
)
```

Additional behavior on top of `__init__`:

1. Checks that `cell_ontology/cell_to_cell_ontology.csv` and `cell_ontology/cl.owl` exist.
2. Computes the expected ontology embedding and description paths from `(agent.model.model_name, embedding.model)`.
3. If `force_regenerate=True`, unlinks both the ontology embeddings NPZ and the descriptions JSON before proceeding.
4. Loads the descriptions JSON (if present) and drops any blank entries so they can be retried.
5. If every CL id has a non-blank description on disk and the embeddings NPZ already exists, returns immediately.
6. Otherwise, generates descriptions only for the missing or blank ids, merges them back into the JSON (still skipping any that came back blank), and rebuilds the ontology embeddings NPZ for all CL ids. For CL ids that still have no description after retries, the raw joined label is used as the embedding text so the NPZ stays aligned; those ids are retried on the next call.

Use this when you want the package to be ready for comparison calls right after construction.

### `compare` (async)

```python
df = await cyto.compare(
    author_labels: list[str],
    algorithms: dict[str, list[str]] | Sequence[tuple[str, list[str]]],
    *,
    run_id: str | None = None,
    metric: str = "cosine_kernel",
    metric_params: dict[str, Any] | None = None,
    min_match_similarity: float = 0.1,
    use_cache: bool = True,
)
```

All arguments after the `*` are keyword-only.

Constraints:

- Every value in `algorithms` must be the same length as `author_labels`. Mismatched lengths raise `ValueError`.
- Algorithm names must be unique. `"author"` is reserved and rejected.
- `run_id` is used as a cache namespace on disk. Reusing the same id reuses cached author and algorithm embeddings when the labels match. If you pass `None`, a UUID of the form `run-<uuid4>` is generated and logged; the same value is written into every result row so you can recover it later.

Call flow:

1. `_embed_user_labels(author_labels, kind="author", identifier="author")` returns a `(N, D)` NumPy array.
2. `_match(author_emb)` returns the best CL id and similarity for each row.
3. `_ensure_similarity()` lazy-loads the OWL file and the ontology embeddings into `OntologySimilarity`.
4. For each `(algo_name, algo_labels)`:
   - `_embed_user_labels(algo_labels, kind="algorithm", identifier=algo_name)`.
   - `_match(algo_emb)`.
   - For each index `i`, `similarity.similarity(author_id, algo_id, metric=..., metric_params=...)` if both ids are present; otherwise `0.0`.
5. Rows are concatenated into a DataFrame with columns from `RESULT_COLUMNS`.

### `compare_anndata` (async)

Pulls label lists out of `adata.obs[target_columns]` for each AnnData object and delegates to `compare`. Skips any object missing `author_column` and warns for missing target columns.

### Cache utilities

```python
cyto.cache_stats(run_id=None)
# {"root", "total_files", "total_size_mb", "stale_files"}

cyto.clear_run(run_id, kind=None, identifier=None) -> int
# deletes user embeddings + descriptions for a run.
# kind=None           -> removes all kinds under the run
# kind="author"       -> removes only the author subtree
# kind="algorithm"    -> removes all algorithm caches under the run
# kind="algorithm", identifier="algo1" -> removes just that algorithm's files

cyto.purge_stale(run_id=None) -> int
# deletes user NPZs that lack the inline `labels` key (legacy format)
```

Scope is `data/user_files/embeddings/` and the matching `descriptions/` subtree (optionally filtered by `run_id`).

### `usage`

`cyto.usage` is an `AgentUsage` accumulating every LLM call made through the instance: request count, input/output/total tokens, and per-tool counts. Use it after `from_config` or `compare` to measure cost.

---

## `compare` call graph

```
CyteOnto.compare
 ├─ _embed_user_labels(author_labels, kind="author", identifier="author")
 │   ├─ storage.load_descriptions(path)          (drop any cached blanks)
 │   ├─ storage.load_user_embeddings(path)       (cache hit only when labels match AND every description is non-blank)
 │   ├─ describe.describe_cells(missing_or_blank) (LLM calls for anything new, plus a 2nd pass for blanks)
 │   │   └─ describe.describe_cell (per label, tenacity-retried agent.run with per-attempt timeout)
 │   ├─ embed.embed_texts(sentences, EmbdConfig) (label text is used as a fallback for still-blank slots)
 │   ├─ storage.save_descriptions                (blanks are filtered out)
 │   └─ storage.save_user_embeddings
 ├─ _match(author_emb)
 │   ├─ storage.load_ontology_embeddings         (cached on the instance)
 │   └─ sklearn.cosine_similarity
 ├─ _ensure_similarity
 │   └─ OntologySimilarity(owl_path, embeddings_path)
 │       ├─ owlready2.get_ontology(...).load()
 │       └─ np.load(ontology_embeddings_npz)
 └─ for each algorithm:
     ├─ _embed_user_labels(algo_labels, kind="algorithm", identifier=algo_name)
     ├─ _match(algo_emb)
     └─ OntologySimilarity.similarity(a_id, g_id, metric, metric_params)
```

---

## Similarity metrics

Implemented in `OntologySimilarity.similarity(id1, id2, metric, metric_params)`.

| Metric                         | Needs embeddings | Needs OWL | Notes                                                              |
|--------------------------------|:----------------:|:---------:|--------------------------------------------------------------------|
| `cosine_kernel` (default)      | yes              | no        | Gaussian hill on raw embedding cosine.                             |
| `cosine_direct`                | yes              | no        | Raw embedding cosine.                                              |
| `path`                         | no               | yes       | `1 / avg(depth(id1, LCA), depth(id2, LCA))`.                       |
| `set:jaccard`                  | no               | yes       | Jaccard of ancestor sets including the term itself.                |
| `set:cosine`                   | no               | yes       | `|A ∩ B| / sqrt(|A| * |B|)`.                                       |
| `set:weighted_jaccard`         | no               | yes       | Jaccard weighted by ancestor depth.                                |
| `weighted:num_ancestors`       | no               | yes       | Weighted jaccard using `1 / (|ancestors(a)| + 1)`.                 |
| `weighted:specificity`         | no               | yes       | Weighted jaccard using ancestor depth divided by term depth.       |
| `weighted:embedding_cosine`    | yes              | yes       | Weighted jaccard using cosine between ancestor and term embeddings. |
| `simple`                       | no               | no        | `difflib.SequenceMatcher` fallback on the CL id strings.           |

`cosine_kernel` supported params (`metric_params`):

| Key        | Default | Description                                  |
|------------|---------|----------------------------------------------|
| `center`   | `1.0`   | Peak of the Gaussian on the cosine axis.     |
| `width`    | `0.25`  | Standard deviation of the Gaussian.          |
| `amplitude`| `1.0`   | Peak value of the hill.                      |

Return value is always in `[0, 1]`. Identical ids short-circuit to `1.0`. On any unexpected failure the code falls back to `simple` with a warning.

---

## LLM description generation

`describe.describe_cell(base_agent, label, use_pubmed=True)`:

1. Builds a dedicated `Agent` with `output_type=CellDescription` and a plain `get_pubmed_abstracts` tool when `use_pubmed=True`.
2. Runs the agent with a directive prompt that specifies each field, length caps, tone, and the treatment of ambiguous labels. No examples are embedded in the prompt to avoid anchoring.
3. Tallies `request/token/tool` usage from the pydantic-ai run into an `AgentUsage`.
4. Returns `(CellDescription, AgentUsage)`.

Retries: up to `RETRY_ATTEMPTS = 4` attempts with exponential backoff (`RETRY_WAIT_MIN = 1 s` to `RETRY_WAIT_MAX = 15 s`) on `ModelHTTPError`, `UnexpectedModelBehavior`, pydantic `ValidationError`, `ConnectionError`, `TimeoutError`, and `asyncio.TimeoutError`. Each attempt is capped at `PER_ATTEMPT_TIMEOUT = 120 s` via `asyncio.wait_for` so a single hung HTTP call cannot stall the batch.

Every failed attempt logs one `WARNING` line via the tenacity `before_sleep` hook, with the label, the attempt number, the underlying exception (HTTP status is surfaced for `ModelHTTPError`), and the next backoff:

```
WARNING: [alveolar macrophage] attempt 2/4 failed: ModelHTTPError(status=429): ... Retrying in 2.0s
```

The `RetryError` that tenacity raises after the final attempt is unwrapped before logging so the original exception class and message appear in the log line.

If all attempts fail, `describe_cell` returns `(CellDescription.blank(label), AgentUsage(...))` and the run continues.

Usage limits default to `request_limit=50, input_tokens_limit=60_000`. Override with the `usage_limits` keyword.

`describe.describe_cells(base_agent, labels, use_pubmed=True, max_concurrent=10, second_pass_wait_seconds=5.0)`:

1. First pass: runs `describe_cell` for every label under an `asyncio.Semaphore(max_concurrent)`. Progress is logged on a 5 percent stride (`Description progress (pass 1): 13/260`).
2. If any slot came back blank, sleeps `second_pass_wait_seconds` and reruns `describe_cell` only for those labels. This gives a transient outage a full second set of 4 retries without requiring the caller to rerun anything.
3. Emits a single end-of-batch summary at `INFO` on full success, or `WARNING` if anything stayed blank (listing the first ten offending labels and truncating the rest).

Tune this by editing the constants at the top of `describe.py` or by passing `second_pass_wait_seconds` explicitly; the other values are module-level for now.

### `CellDescription` schema

| Field                | Type    | Notes                                                              |
|----------------------|---------|--------------------------------------------------------------------|
| `initialLabel`       | `str`   | Copied verbatim from the input, including any synonym list.        |
| `descriptiveName`    | `str`   | One noun phrase, at most 12 words.                                 |
| `function`           | `str`   | Up to 3 sentences, soft cap 60 words.                              |
| `diseaseRelevance`   | `str`   | Up to 3 sentences, soft cap 60 words. `"Not established"` allowed. |
| `developmentalStage` | `str`   | One sentence, soft cap 30 words. `"Not established"` allowed.      |

`model_config = {"extra": "ignore"}` so legacy JSON files with extra fields (for example `markerGenes`) still deserialise, though the extra content is dropped.

`CellDescription.to_sentence()` produces the flat text that is fed to the embedding model:

```
<initialLabel> is <descriptiveName>. Function: <function> Disease relevance: <diseaseRelevance> Developmental stage: <developmentalStage>
```

`CellDescription.blank(label)` returns a zeroed instance used as a fallback after an unrecoverable LLM failure.

### PubMed tool

`describe.get_pubmed_abstracts(query, max_results=5)` hits NCBI eutils, returns a `list[str]`, and swallows all errors (returning `[]`). `NCBI_API_KEY` raises the rate limit when set.

---

## Embedding generation

`embed.embed_texts(texts, EmbdConfig)`:

- Dispatches to a per-provider URL and request body via `_PROVIDER_URL`, `_headers`, `_build_payload`, `_extract_embedding`.
- Sends one request per text to avoid provider-side batch averaging.
- Bounds concurrency with `asyncio.Semaphore(cfg.maxConcurrent)`.
- Each request is retried up to 3 times with exponential backoff (`tenacity`).
- Logs an initial `INFO` banner (`Embedding N texts (provider=..., model=..., concurrency=...)`) and then a progress line on a 5 percent stride (`Embedding progress: 13/260`).
- Returns a `(N, D) float32` `numpy.ndarray` or `None` if any text failed after retries.

Supported providers and URLs:

| Provider     | Endpoint                                                                 |
|--------------|--------------------------------------------------------------------------|
| `deepinfra`  | `https://api.deepinfra.com/v1/openai/embeddings`                         |
| `openrouter` | `https://openrouter.ai/api/v1/embeddings`                                |
| `ollama`     | `http://localhost:11434/api/embed`                                       |
| `openai`     | `https://api.openai.com/v1/embeddings`                                   |
| `google`     | `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent` |
| `together`   | `https://api.together.xyz/v1/embeddings`                                 |

Together AI also supports pydantic-ai for LLM calls via an OpenAI-compatible interface. See [docs.together.ai/docs/pydanticai](https://docs.together.ai/docs/pydanticai) if you want to drive the description agent through Together; it lives on the caller side and is orthogonal to the embedding provider above.

---

## Storage

Two NPZ layouts are in use.

### Ontology NPZ (`storage.save_ontology_embeddings` / `load_ontology_embeddings`)

Keys: `embeddings`, `ontology_ids`, `metadata`.

`ontology_ids` is the array of CL ids, one per row of `embeddings`.

### User NPZ (`storage.save_user_embeddings` / `load_user_embeddings`)

Keys: `embeddings`, `labels`, `metadata`.

`labels` holds the original user-provided labels, one per row. On load, the list is compared to the labels the caller wants. If they match exactly (same order and contents), the embeddings are reused; otherwise the file is treated as stale and a fresh generation is triggered.

Files that lack the `labels` key (the legacy sidecar format) are rejected by `load_user_embeddings`. `CyteOnto.purge_stale()` deletes them.

### Descriptions JSON

```json
{
  "<key>": {
    "initialLabel": "...",
    "descriptiveName": "...",
    "function": "...",
    "diseaseRelevance": "...",
    "developmentalStage": "..."
  }
}
```

- For ontology descriptions, `<key>` is the CL id.
- For user descriptions, `<key>` is the label the caller passed to `compare`.

---

## On-disk layout

Set via `PathConfig(data_dir=..., user_dir=...)`:

```
<data_dir>/
├── cell_ontology/
│   ├── cell_to_cell_ontology.csv      shipped
│   └── cl.owl                         shipped
└── embedding/
    ├── cell_ontology/
    │   └── embeddings_<text>_<embd>.npz       generated once per model pair
    └── descriptions/
        └── descriptions_<text>.json           generated once per text model

<user_dir>/                             defaults to <data_dir>/user_files
├── embeddings/
│   └── <run_id>/
│       ├── author/author_embeddings_<text>_<embd>.npz
│       └── algorithm/<algo_name>_embeddings_<text>_<embd>.npz
└── descriptions/
    └── <run_id>/
        ├── author/author_descriptions_<text>.json
        └── algorithm/<algo_name>_descriptions_<text>.json
```

Filename rules (`paths._clean_model`, `paths._clean_identifier`):

- Model names preserve case and dots. `/` and `:` become `-`, space becomes `_`.
  Example: `moonshotai/kimi-k2.5` becomes `moonshotai-kimi-k2.5`.
- Identifiers (`run_id`, algorithm keys) also replace `.` with `_` to keep directory names conservative.
  Example: `sample.run.001` becomes `sample_run_001`.

---

## Caching rules

| Artifact                          | Trigger for regeneration                                                |
|-----------------------------------|-------------------------------------------------------------------------|
| Ontology descriptions JSON        | File missing, any CL id missing from the JSON, any cached entry is blank, or `force_regenerate=True`. |
| Ontology embeddings NPZ           | File missing, any description was regenerated on this call, or `force_regenerate=True`. |
| User author/algorithm embeddings  | File missing, `labels` array inside the NPZ differs from the request, or any description for a requested label is missing or blank. |
| User author/algorithm descriptions| Per-label: any label missing from the existing JSON or cached as blank is regenerated; non-blank existing entries are kept. |

Per-label description caching means that adding one new label to a run does not recompute descriptions for existing labels; only the new label hits the LLM.

Blank descriptions (the fallback returned by `describe_cell` after an unrecoverable LLM failure) are never written to the JSON cache. On the next call, those labels are treated as missing and retried. The accompanying NPZ still covers them using the raw label text as the embedding input so the array stays aligned with the requested label list.

When `use_cache=False` is passed to `compare`, all cache lookups are skipped and everything is regenerated.

`force_regenerate=True` on `from_config` unlinks both the ontology NPZ and the ontology descriptions JSON before regenerating, so the description set stays in sync with the embedding set.

---

## Error behavior

- Empty embedding API key for a non-`ollama` provider: `CyteOnto.__init__` raises `ValueError`.
- Core ontology files missing: `from_config` raises `FileNotFoundError`.
- Ontology embedding generation failure: `from_config` raises `RuntimeError`.
- User embedding generation failure: `compare` raises `RuntimeError` with the offending identifier.
- Label length mismatch between author and algorithm lists: `compare` raises `ValueError`.
- Duplicate algorithm name or an algorithm named `"author"`: `compare` raises `ValueError`.
- Ontology match below `min_match_similarity`: CL id stored as `None`, `similarity_method` becomes `partial_match` or `no_matches`.
- OWL class not found for a CL id during hierarchy metrics: falls back to `simple` similarity with a warning.
- Per-label LLM failure: the label gets a blank `CellDescription` after 4 attempts and a second-pass retry; the run continues. The blank is not persisted to the descriptions JSON and will be retried on the next call. The corresponding embedding row is computed using the raw label text so the NPZ stays aligned with the requested label list.
- Per-request embedding failure: the whole `embed_texts` call returns `None` after 3 tenacity retries, which surfaces as a `RuntimeError` from `_embed_user_labels`.

---

## Extending

- **New provider**: add a URL to `_PROVIDER_URL` in `embed.py`, extend the `EmbdProvider` `Literal` in `models.py`, and adjust `_headers` / `_build_payload` / `_extract_embedding` if the request or response shape differs.
- **New similarity metric**: implement a helper in `OntologySimilarity`, add a branch in `OntologySimilarity.similarity`, and update the table above.
- **Alternative prompt**: edit `describe._build_prompt`. The existing prompt lists every `CellDescription` field with a soft length cap and a neutral tone; keep the same structure to avoid anchoring bias.
- **Different ontology file**: pass a custom `data_dir`. The package expects the two files under `<data_dir>/cell_ontology/` to be named exactly `cl.owl` and `cell_to_cell_ontology.csv`.

---

## Minimum dependency set

Already declared in the project `pyproject.toml`:

- `pydantic`, `pydantic-ai`
- `aiohttp`, `tenacity`
- `numpy`, `pandas`, `scikit-learn`
- `owlready2`
- `loguru`, `python-dotenv`, `tqdm`
- `requests` (PubMed tool)
