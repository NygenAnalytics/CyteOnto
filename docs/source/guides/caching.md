---
description: How CyteOnto caches the descriptions and embeddings of the Cell Ontology terms and of your labels, how run IDs name the caches of a comparison, and how to inspect, reuse, regenerate and delete them.
---

(caching)=
# Caching and runs

CyteOnto saves what it gets from the models, so that it asks for each description and
embedding only once. It keeps two kinds of caches:

- **Ontology caches**, in `data_dir`: the descriptions of the Cell Ontology terms for
  each LLM, and their embeddings for each model pair. All runs share them.
  `python -m cyteonto.setup` downloads those of the precomputed model pairs, and
  `from_config()` creates missing ones. After an embedding failover, `compare()` also
  creates those of the fallback embedding model.
- **Run caches**, in `user_dir`, which is `<data_dir>/user_files` by default: how your
  labels were split, their descriptions and their embeddings, under the `run_id` of
  each `compare()` call.

## Run IDs

The `run_id` of `compare()` names the caches of a comparison and fills the `run_id`
column of the results.

- **Reuse a run ID** to rerun a comparison: with the same labels, `compare()` loads the
  run caches and calls no model.
- **Use a new run ID** for a separate comparison, so that its caches start empty.
- **Omit it**, and CyteOnto generates one of the form `run-<uuid4>`, logs it and writes
  it in the `run_id` column, so you can reuse it from `df["run_id"].iloc[0]`.

In file paths, `/` and `:` in run IDs and algorithm names become `-`, and spaces and
`.` become `_`: `liver.v1` is stored as `liver_v1`. The algorithm name `author` is
reserved for the author labels.

## What is reused

| Cache | Reused |
|---|---|
| How labels were split | Per label, for all the labels of the run, author and algorithm alike |
| Label descriptions | Per label, separately for the authors and for each algorithm |
| Label embeddings | When the authors' or algorithm's distinct labels are exactly those of the saved file, in the same order, and all have descriptions |
| Ontology descriptions | Per term: only the terms without a description are described |
| Ontology embeddings | When the file exists and every term already has a description; otherwise all 2,902 terms are embedded again |

So when you add a label to a run, only the new label is split and described, but all the
labels of that side are embedded again. Labels are lowercased first, so `Plasma cell`
and `plasma cell` share their cache entries.

A label that the LLM could not describe after all retries is not cached: its embedding
is made from the label text, and the next call with the run tries to describe it again.
The same holds for ontology terms in `from_config()`, and for labels whose split failed,
except a split that exceeds the usage limits: that label stays whole, and this is cached,
since another try would exceed the limits again.

When labels or terms get new descriptions, the embedding files made before are stale, for
every embedding model: CyteOnto embeds the descriptions again instead of reusing those
files. A stale file stays on disk until a new one replaces it.

## Regenerate caches

- `compare(..., use_cache=False)` ignores the run caches, asks the models again, and
  overwrites the cached results of the labels it was given. A label that gets no new
  description or split keeps its cached one.
- `from_config(..., force_regenerate=True)` deletes the ontology descriptions of the
  LLM and their embeddings with the primary and fallback embedding models. It then
  generates the descriptions and the primary model's embeddings again for all 2,902
  terms; the fallback model's are made again at the next failover. With the precomputed
  model pairs, run `python -m cyteonto.setup --force` instead, which downloads them
  again. The ontology embeddings of other embedding models are then stale, and are made
  again by `from_config()` with such a model, or at a failover to it.

## Inspect and delete run caches

`cache_stats()` summarizes the label embedding files of all runs, or of one run:

```python
cyto.cache_stats(run_id="quickstart")
```

It returns a dict with `root`, the directory it searched, `total_files` and
`total_size_mb`, the number and size of the embedding files, and `stale_files`, the
files that cannot be loaded.

`clear_run()` deletes the descriptions and embeddings of a run and returns the number of
files deleted:

```python
# Everything of the run.
cyto.clear_run("quickstart")

# Only the author labels, or only the algorithms.
cyto.clear_run("quickstart", kind="author")
cyto.clear_run("quickstart", kind="algorithm")

# One algorithm, for the model pair of this instance.
cyto.clear_run("quickstart", kind="algorithm", identifier="method_a")
```

`clear_run()` keeps how the labels were split. Delete
`<user_dir>/decompositions/<run_id>/` to remove those too.

`purge_stale()` deletes the label embedding files that cannot be loaded, such as files
written by older versions of CyteOnto without the labels inside, from all runs or from
one run:

```python
cyto.purge_stale(run_id="quickstart")
```

## Choose where the caches go

`data_dir` defaults to the `data` directory inside the installed package and `user_dir`
to `<data_dir>/user_files`. Pass both to keep the ontology files in one place and the
run caches of a project in another:

```python
from pathlib import Path

cyto = await CyteOnto.from_config(
    agent=agent,
    embedding=embedding,
    llm=llm,
    data_dir=Path.home() / "cyteonto-data",
    user_dir=Path("cyteonto-runs"),
)
```

`data_dir` must contain the files that `python -m cyteonto.setup --data-dir` downloads.
[Data files and formats](../reference/files.md) shows the layout of both directories
and the formats of the files.
