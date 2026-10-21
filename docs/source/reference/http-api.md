---
description: Reference for the hosted CyteOnto HTTP API at cyteonto.nygen.io, with every field of a comparison request, the job status, the result formats and the error responses.
---

(http-api)=
# HTTP API

The hosted API runs comparisons at `https://cyteonto.nygen.io`. Requests and responses
are JSON. The service also serves interactive documentation at
[`/docs`](https://cyteonto.nygen.io/docs) and its OpenAPI schema at
[`/openapi.json`](https://cyteonto.nygen.io/openapi.json).

| Method and path | Purpose |
|---|---|
| `POST /compare` | Submit a comparison job |
| `GET /status/{runId}` | Get the state of a job |
| `GET /result/{runId}` | Download the results of a completed job |
| `GET /health` | Check that the service is up |

A job runs in the background: submit it, poll its status until it is `completed` or
`failed`, then download the results. The
[hosted API quickstart](../getting-started/quickstart-api.md) shows these steps with
curl and Python.

(http-api-compare)=
## `POST /compare`

Submit the author labels and the labels of one or more algorithms. Only `authorLabels`
and `algorithms` are required; the other fields select the models and the options of
the comparison, as the arguments of
[`CyteOnto.compare()`](api/cyteonto.html#cyteonto.CyteOnto.compare) do in Python.

| Field | Type | Default | Description |
|---|---|---|---|
| `authorLabels` | array of strings | Required | The author labels, one per item. Must not be empty |
| `algorithms` | object | Required | Maps each algorithm's name to its labels, one per author label, in the same order. Needs at least one algorithm; `author` is not a valid name |
| `llmProvider` | string | `nebius` | The LLM provider: `nebius`, `fireworks`, `openrouter`, `together` or `openai` |
| `llmModel` | string | `moonshotai/Kimi-K2.6` | The LLM, as the provider names it |
| `llmApiKey` | string or null | `null` | Your API key for `llmProvider`. When omitted, the service uses its own key for the provider, and the job fails if it has none |
| `embeddingProvider` | string | `nebius` | The embedding provider: `nebius`, `deepinfra`, `fireworks`, `google`, `openai`, `openrouter` or `together` |
| `embeddingModel` | string | `Qwen/Qwen3-Embedding-8B` | The embedding model, as the provider names it |
| `embeddingApiKey` | string or null | `null` | Your API key for `embeddingProvider`. When omitted, the service uses its own key for the provider, and the job fails if it has none |
| `embeddingModelSettings` | object or null | `null` | Fields added to each embedding request body, for `openrouter` only. `null` routes OpenRouter requests to DeepInfra first, and `{}` sends no routing |
| `embeddingMaxConcurrent` | integer, at least 1 | `100` | Maximum number of embedding requests at the same time |
| `maxDescriptionConcurrency` | integer, at least 1 | `100` | Maximum number of LLM requests at the same time |
| `usePubmedTool` | boolean | `false` | Let the LLM search PubMed while it describes a label |
| `reasoning` | boolean | `false` | Leave the model's reasoning as the provider sets it; when `false`, the requests ask the provider to turn it off |
| `metric` | string | `cosine_kernel` | The [similarity metric](../guides/similarity-metrics.md) |
| `metricParams` | object or null | `null` | The `center`, `width` and `amplitude` of `cosine_kernel` |
| `minMatchSimilarity` | number, 0 to 1 | `0.1` | The [match threshold](../guides/similarity-metrics.md#similarity-threshold) |
| `compoundScoring` | string | `max` | How [compound labels](../guides/compound-labels.md) are scored: `max` or `hungarian_mean` |
| `useCache` | boolean | `true` | Reuse the caches of the run. Each job has a new run ID, so this only reuses the splits of labels that the authors and the algorithms of the job share |

The default models match the precomputed ontology files. With another LLM or embedding
model, the job first describes or embeds the 2,902 Cell Ontology terms for that model
pair, which takes much longer. The service calls the LLM through the provider's
OpenAI-compatible API. If descriptions fail, the service retries them with its fallback
LLM, Kimi K2.6 on Fireworks, and the status reports it. It also falls back to
Qwen3-Embedding-8B on OpenRouter when embeddings fail; the first job that does so also
embeds the 2,902 ontology descriptions with that model, which takes longer. The schema
also accepts `ollama` as `embeddingProvider`, but the service has no Ollama server: each
such job first tries to reach it for every ontology term and then switches to the
fallback model, so do not use it with the hosted API. See
[Choose the embedding provider](../guides/models-and-providers.md).

The response holds the new job's run ID:

```json
{"runId": "run-<uuid>", "state": "queued"}
```

### Request options

Options go in the same body as the labels. This request scores compound labels with
`hungarian_mean`, leaves weak matches unmatched and widens the kernel of the metric:

```bash
curl -sS -X POST "https://cyteonto.nygen.io/compare" \
  -H 'Content-Type: application/json' \
  -d '{
    "authorLabels": ["AT2 cell-plasma cell doublet", "regulatory T cell"],
    "algorithms": {"method_a": ["plasma cell", "Treg"]},
    "compoundScoring": "hungarian_mean",
    "minMatchSimilarity": 0.5,
    "metricParams": {"width": 0.5}
  }'
```

To send your own API keys, build the body with jq, which quotes the values of shell
variables correctly. Inside the single quotes of `-d '...'`, the shell would send
`$NEBIUS_API_KEY` as literal text:

```bash
jq -n --arg key "$NEBIUS_API_KEY" '{
  authorLabels: ["alveolar macrophage", "regulatory T cell"],
  algorithms: {method_a: ["lung macrophage", "Treg"]},
  llmApiKey: $key,
  embeddingApiKey: $key
}' | curl -sS -X POST "https://cyteonto.nygen.io/compare" \
  -H 'Content-Type: application/json' -d @-
```

## `GET /status/{runId}`

Returns the state of a job.

| Field | Type | Description |
|---|---|---|
| `runId` | string | The run ID |
| `state` | string | `queued`, `running`, `completed` or `failed` |
| `createdAt` | string | When the job was submitted, as an ISO 8601 UTC timestamp |
| `startedAt` | string or null | When the job started |
| `completedAt` | string or null | When the job completed or failed |
| `error` | string or null | For a failed job, the exception that stopped it, as `<ExceptionClass>: <message>` |
| `numAuthorLabels` | integer | The number of author labels |
| `numAlgorithms` | integer | The number of algorithms |
| `numRows` | integer or null | For a completed job, the number of result rows |
| `resultCsvPath` | string or null | For a completed job, the path of the CSV result on the service's storage |
| `resultJsonPath` | string or null | For a completed job, the path of the JSON result on the service's storage |
| `modelPairUsage` | object or null | Once the job has run: `llmTier`, `primary`, or `mixed` when the fallback LLM was asked to describe labels; `embeddingTier`, the models that made the label embeddings that were matched, `primary`, `fallback`, or `mixed` for both; and the `llmProvider`, `llmModel`, `embeddingProvider` and `embeddingModel`. When a tier is not `primary`, its fields name the fallback model |

A job normally moves from `queued` to `running` and ends as `completed` or `failed`. A
job that stops unexpectedly can keep its last state, so stop polling after a time
limit.

## `GET /result/{runId}`

Returns the results of a completed job. The `format` query parameter selects the format:

- `format=json`, the default, returns a JSON array with one object per row.
- `format=csv` returns a CSV file named `<runId>.csv`.

Both have the columns described in [Read the results](../guides/results.md).

## `GET /health`

Returns `{"ok": true}` when the service is up.

## Errors

Errors are JSON objects with a `detail` field.

| Status | When |
|---|---|
| 400 | `POST /compare`: `authorLabels` or `algorithms` is empty, or an algorithm's labels differ in number from the author labels. `GET /result`: `format` is neither `json` nor `csv` |
| 404 | No job has this run ID |
| 409 | `GET /result`: the job has not completed |
| 422 | `POST /compare`: the body does not match the schema, such as a missing required field, an unknown field, a value of the wrong type, or a value out of range; `detail` lists the problems |
| 500 | `GET /result`: the result of a completed job is missing |

Other problems show up while the job runs. A job fails, with the reason in `error`, for
example when an algorithm is named `author`, when no API key is available for a
provider, or when the labels cannot be embedded. The service does not check your API
keys when you submit a job. If the provider rejects an LLM key, the fallback LLM may
describe the labels instead and the job completes, so check `modelPairUsage`.

## Privacy

The service has no accounts: anyone who has a run ID can read the job's status and
results. Run IDs are random, but share them only with people who may see your labels.
The service sends your labels to the LLM and embedding providers of the job, and may
send them to its fallback providers, Fireworks and OpenRouter, even when you select
others. With `usePubmedTool`, the model's search terms go to NCBI. The service keeps the
status, the results and the label splits, descriptions and embeddings of each job.
