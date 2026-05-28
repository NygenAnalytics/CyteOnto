# File Management

CyteOnto stores ontology artifacts under `data_dir` and per-run user artifacts under `user_dir` (default: `<data_dir>/user_files`). Paths are resolved by `PathConfig` in `cyteonto/paths.py`. See [cyteonto/README.md](../cyteonto/README.md#on-disk-layout) for storage key layouts inside NPZ and JSON files.

## Directory structure

Default roots: `cyteonto/data/` for shipped and generated ontology data, `cyteonto/data/user_files/` for comparison caches.

```
<data_dir>/
├── cell_ontology/
│   ├── cl.owl                              # shipped
│   └── cell_to_cell_ontology.csv           # shipped
└── embedding/
    ├── cell_ontology/
    │   └── embeddings_<text_model>_<embd_model>.npz
    └── descriptions/
        └── descriptions_<text_model>.json

<user_dir>/                                 # default: <data_dir>/user_files
├── embeddings/
│   └── <run_id>/
│       ├── author/
│       │   └── author_embeddings_<text>_<embd>.npz
│       └── algorithm/
│           ├── <algo1>_embeddings_<text>_<embd>.npz
│           └── <algo2>_embeddings_<text>_<embd>.npz
└── descriptions/
    └── <run_id>/
        ├── author/
        │   └── author_descriptions_<text>.json
        └── algorithm/
            ├── <algo1>_descriptions_<text>.json
            └── <algo2>_descriptions_<text>.json
```

`<run_id>` and algorithm names are passed through `_clean_identifier` (e.g. `sample.run.001` becomes `sample_run_001`). Model names use `_clean_model` (slashes and colons become hyphens; case and dots are preserved).

## File naming

### Ontology (generated once per model configuration)

| Artifact | Pattern | Example |
|----------|---------|---------|
| Descriptions | `descriptions_<llmKey>.json` | `descriptions_together_moonshotai-Kimi-K2.6.json` |
| Embeddings | `embeddings_<llmKey>_<embdKey>.npz` | `embeddings_together_moonshotai-Kimi-K2.6_openrouter_qwen-qwen3-embedding-8b.npz` |

### User files (per `run_id`, kind, and identifier)

| Artifact | Pattern |
|----------|---------|
| Author descriptions | `author_descriptions_<llmKey>.json` |
| Author embeddings | `author_embeddings_<llmKey>_<embdKey>.npz` |
| Algorithm descriptions | `<identifier>_descriptions_<llmKey>.json` |
| Algorithm embeddings | `<identifier>_embeddings_<llmKey>_<embdKey>.npz` |

`kind` is either `author` or `algorithm`. For author rows, `identifier` is always `author`.

## Custom paths

```python
from cyteonto import LlmConfig

cyto = await CyteOnto.from_config(
    agent=agent,
    embedding=embd,
    llm=LlmConfig(provider="together", model="moonshotai/Kimi-K2.6"),
    data_dir="/path/to/data",
    user_dir="/path/to/user_files",  # optional; defaults to data_dir/user_files
)
```

Custom `data_dir` must still contain `cell_ontology/cl.owl` and `cell_ontology/cell_to_cell_ontology.csv` with those exact filenames.

## Caching behavior

| Artifact | Reused when |
|----------|-------------|
| Ontology descriptions | File exists, every CL id has a non-blank entry, and `force_regenerate=False` |
| Ontology embeddings | NPZ exists and no descriptions were regenerated on this `from_config` call |
| User descriptions | Label already in JSON with a non-blank `CellDescription` |
| User embeddings | NPZ exists, embedded `labels` array matches the request exactly, and every label has a non-blank description |

Blank LLM failures are not written to JSON; those labels are retried on the next call. Embeddings for those positions may still use the raw label text so array shape stays aligned.

Per-label caching means adding one new label to a run does not re-describe existing labels.

Set `use_cache=False` on `compare` to bypass all of the above for that invocation.

## Cache utilities

### Statistics

```python
stats = cyto.cache_stats()
print(stats["total_files"], stats["total_size_mb"], stats["stale_files"])

# Restrict to one run
stats = cyto.cache_stats(run_id="sample_run_001")
```

`stale_files` counts NPZ files that fail validation (for example legacy files missing the inline `labels` key).

### Remove a run

```python
cyto.clear_run("sample_run_001")                                    # entire run
cyto.clear_run("sample_run_001", kind="author")                     # author only
cyto.clear_run("sample_run_001", kind="algorithm")                  # all algorithms
cyto.clear_run("sample_run_001", kind="algorithm", identifier="algo1")
```

Returns the number of files removed.

### Purge invalid NPZ files

```python
removed = cyto.purge_stale()                    # all runs
removed = cyto.purge_stale(run_id="sample_run_001")
```

Deletes user embedding NPZ files that `load_user_embeddings` cannot validate.

### Manual cleanup

```bash
# Inspect caches (default package data dir)
tree cyteonto/data/user_files/

# Remove one run
rm -rf cyteonto/data/user_files/embeddings/my_run_id/
rm -rf cyteonto/data/user_files/descriptions/my_run_id/

du -sh cyteonto/data/user_files/embeddings/*/
```

## Run IDs

`run_id` is the cache namespace for a comparison:

```python
df = await cyto.compare(
    author_labels=[...],
    algorithms={"method_a": [...], "method_b": [...]},
    run_id="liver_scrna_2024_v1",
)
```

- **Reuse** the same `run_id` when rerunning with identical label lists to avoid redundant API calls.
- **Choose a new `run_id`** when labels or algorithms change enough that you want a clean cache tree.
- **Omit `run_id`** to auto-generate `run-<uuid4>`; recover it from logs or `df["run_id"].iloc[0]`.

Naming tips: use descriptive, filesystem-safe strings (`brain_cohort_v2`, `2024-05-rerun`). Avoid reserving the name `author` for algorithm keys.

## LLM usage tracking

`cyto.usage` is an `AgentUsage` object that accumulates token and request counts across `from_config` and `compare` for cost monitoring.

## Related documentation

- [WORKFLOW.md](WORKFLOW.md) — Process flow and diagrams
- [cyteonto/README.md](../cyteonto/README.md) — Full API, metrics, and error behavior
