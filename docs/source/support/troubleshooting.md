---
description: Fixes for common CyteOnto problems, from installing and downloading the ontology files to API keys, model names, async calls, failed embeddings, unexpected scores and errors of the hosted API.
---

(troubleshooting)=
# Troubleshooting

## Setup and installation

**`ImportError: attempted relative import with no known parent package` when you run
`python cyteonto/setup.py`.** The setup script is a module of the package. Run it with
`python -m cyteonto.setup`.

**`FileNotFoundError: Core ontology files missing under <data_dir>/cell_ontology/`.**
`from_config()` needs `cell_to_cell_ontology.csv`, `cell_to_cell_ontology_enriched.csv`
and `cl.owl` in `<data_dir>/cell_ontology/`. Run `python -m cyteonto.setup`, with `--data-dir` if you pass
a `data_dir` to CyteOnto, and pass the same directory to both.

**`from_config()` runs for a long time and sends thousands of requests.** Your model
pair has no ontology files yet, so CyteOnto describes or embeds all 2,902 terms. If you
meant to use a precomputed pair, check that the provider and model strings match
[the precomputed pairs](../guides/models-and-providers.md#models-precomputed) exactly,
including their case. `EmbdConfig()` without arguments matches neither.

**`Descriptions at <path> are not schema v3` in the log.** The file comes from an older
version of CyteOnto and is ignored. Delete it; for the ontology files, run
`python -m cyteonto.setup --force`.

## Models and API keys

**`ValueError: Embedding provider '<provider>' requires an API key`.** Pass
`EmbdConfig(apiKey=...)`, or set the provider's variable, such as `NEBIUS_API_KEY`, or
`EMBEDDING_MODEL_API_KEY`. Set `EMBEDDING_MODEL_API_KEY` before you import `cyteonto`; see
[Environment and limits](../reference/configuration.md).

**`ValueError: LlmConfig.model (...) does not match agent.model.model_name (...)`.**
`LlmConfig.model` must be the model name of the agent, as `agent.model.model_name`
reports it, so that the caches belong to the model in use.

**`RuntimeError: Failed to embed labels for '<identifier>'`, `Failed to generate
ontology embeddings` or `Failed to embed the ontology descriptions with <model>`.** An
embedding request failed after its retries, and the log has an error for each failure.
The last message means that the fallback model failed while CyteOnto embedded the
ontology for it at a failover; the instance keeps the primary model. Check the API
key, the model name and the provider, for example with one request to the provider's
endpoint, or add a `fallback_embedding`; see
[Choose the embedding provider](../guides/models-and-providers.md).

**`RuntimeError: Failed to save the ontology embeddings to <path>`.** CyteOnto could not
write the ontology embeddings in `data_dir`, for example because the directory is
read-only or the disk is full; the log has the cause. At a failover, the instance keeps
the primary model. Make `data_dir` writable, or free disk space.

**`FileNotFoundError: Ontology embeddings not found at <path>. Run
CyteOnto.from_config(...) first.`** The ontology embeddings of the model pair are
missing. Create the instance with `from_config()` rather than `CyteOnto()`.

**`FileNotFoundError: No ontology descriptions for <LLM> to embed with <model>`.** An
embedding request failed, and at the failover CyteOnto found neither ontology
embeddings of the fallback model nor ontology descriptions to make them from. The
instance keeps the primary model. Create the instance with `from_config()`, which loads
or generates the descriptions; see
[Fall back to other models](../guides/models-and-providers.md#models-fallback).

**Many labels end up without a description.** The log shows a warning for each failed
attempt, with the HTTP status, and a summary of the labels that stayed blank. Rate
limits are the most common cause: lower `max_description_concurrency`, or add a
`fallback_agent`. The undescribed labels are retried by the next call with the same
`run_id`.

## Running comparisons

**`SyntaxError: 'await' outside function` in a script.** `from_config()`, `compare()` and
`compare_anndata()` are coroutines. Wrap the calls in an `async` function and run it
with `asyncio.run()`, as in [Run it as a script](../getting-started/quickstart-python.md#quickstart-python-script).

**`RuntimeError: asyncio.run() cannot be called from a running event loop` in a
notebook.** Jupyter already runs an event loop, so `await` the calls directly instead.

**`ValueError: Label length mismatch for '<algorithm>'`.** Each algorithm needs one label
per author label, in the same order.

**`ValueError: Duplicate algorithm name` from `compare_anndata()` with several AnnData
objects.** Each object adds the same algorithm names. Compare one object per call; see
[Labels in AnnData](../guides/anndata.md).

**The log is too long.** CyteOnto logs at the `DEBUG` level by default. Set
`CYTEONTO_LOG_LEVEL=INFO` or `WARNING` before you import `cyteonto`.

## Unexpected results

**Most rows are `partial_match` or `no_matches`.** `min_match_similarity` is too high for
your labels; see [The match threshold](../guides/similarity-metrics.md#similarity-threshold).

**A broad label and a specific one score low, such as `macrophage` and `alveolar
macrophage`.** The default metric compares the descriptions of the two terms, not their
places in the hierarchy. Try a hierarchy metric, such as `set:jaccard`; see
[Similarity metrics](../guides/similarity-metrics.md).

**A label matched an unexpected term.** Look at its `*_ontology_name` and
`*_embedding_similarity` columns. Abbreviations and lab-specific names can be described
as something else; a more explicit label, or a synonym list separated by semicolons,
helps.

**Labels such as `nan` or `none` in the results.** Missing values in `adata.obs` were
converted to strings; drop or fill them first, as shown in
[Labels in AnnData](../guides/anndata.md).

**The scores look like string similarities.** An unknown metric name, or a term that is
not in `cl.owl`, falls back to the `simple` metric, which compares the ID strings. Check
the spelling of `metric` and the log.

## Hosted API

**`422 Unprocessable Entity`.** The request body does not match the schema; `detail`
lists each problem, such as an unknown field or a value out of range. Field names are
case-sensitive: `authorLabels`, not `author_labels`.

**`400 Bad Request` from `POST /compare`.** `authorLabels` or `algorithms` is empty, or an
algorithm's labels differ in number from the author labels; `detail` says which.

**`409 Conflict` from `GET /result`.** The job has not completed: poll `/status` until its
state is `completed`. A failed job has no results, so `/result` keeps returning 409 for
it.

**The job failed.** The `error` field of the status holds the exception. See
[HTTP API errors](../reference/http-api.md#errors).

If your problem is not listed, open an issue in
[NygenAnalytics/CyteOnto](https://github.com/NygenAnalytics/CyteOnto/issues) with the
error message and the log.
