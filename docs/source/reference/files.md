---
description: The directory layout of CyteOnto's data and run caches, how model keys, run IDs and algorithm names become file names, and the formats of the mapping CSV, the description and decomposition JSON files and the embedding NPZ files.
---

(files)=
# Data files and formats

CyteOnto keeps the Cell Ontology and its ontology caches in `data_dir`, and the caches of
comparison runs in `user_dir`. `PathConfig` in `cyteonto/paths.py` resolves the paths,
except that of the label splits, which `CyteOnto` builds in the same way.

## Data directory

`data_dir` defaults to the `data` directory inside the installed package:

```text
<data_dir>/
├── cell_ontology/
│   ├── cell_to_cell_ontology.csv
│   ├── cell_to_cell_ontology_enriched.csv
│   └── cl.owl
└── embedding/
    ├── cell_ontology/
    │   └── embeddings_<llmKey>_<embdKey>.npz
    └── descriptions/
        └── descriptions_<llmKey>.json
```

`from_config()` requires the three files in `cell_ontology/`. Lookups use the enriched
CSV.

## Run directory

`user_dir` defaults to `<data_dir>/user_files`:

```text
<user_dir>/
├── decompositions/
│   └── <run_id>/
│       └── decompositions_<llmKey>.json
├── descriptions/
│   └── <run_id>/
│       ├── author/
│       │   └── author_descriptions_<llmKey>.json
│       └── algorithm/
│           └── <algorithm>_descriptions_<llmKey>.json
└── embeddings/
    └── <run_id>/
        ├── author/
        │   └── author_embeddings_<llmKey>_<embdKey>.npz
        └── algorithm/
            └── <algorithm>_embeddings_<llmKey>_<embdKey>.npz
```

## File names

`<llmKey>` and `<embdKey>` come from
[`ModelArtifactKey.filename_segment()`](api/configuration.html#cyteonto.ModelArtifactKey.filename_segment):
`{provider}_{company}-{modelName}`, where the provider is lowercased, `company` is the
part of the model ID before the first `/` and `modelName` the part after it. A model ID
without `/` gives both. In the result, `/` and `:` become `-` and spaces become `_`; case
and dots are kept.

| Provider | Model | Key |
|---|---|---|
| `nebius` | `moonshotai/Kimi-K2.6` | `nebius_moonshotai-Kimi-K2.6` |
| `nebius` | `Qwen/Qwen3-Embedding-8B` | `nebius_Qwen-Qwen3-Embedding-8B` |
| `fireworks` | `accounts/fireworks/models/kimi-k2p6` | `fireworks_accounts-fireworks-models-kimi-k2p6` |
| `openrouter` | `Qwen/Qwen3-Embedding-8B` | `openrouter_Qwen-Qwen3-Embedding-8B` |
| `openai` | `text-embedding-3-small` | `openai_text-embedding-3-small-text-embedding-3-small` |

In run IDs and algorithm names, `/` and `:` become `-`, and spaces and `.` become `_`:
`liver.v1/batch 2` becomes `liver_v1-batch_2`.

## Ontology mapping

`cell_to_cell_ontology.csv` has the columns `ontology_id` and `label`, with one row per
name or synonym of a term. The first row of a term holds the name that CyteOnto shows for
it, which is the term's Cell Ontology name for nearly all terms. The shipped file has
6,156 rows for 2,902 terms. `cell_to_cell_ontology_enriched.csv` adds `label_normalized`,
the lowercase label, and drops the rows that repeat a term's `label_normalized`.

```text
ontology_id,label,label_normalized
CL:0000000,cell,cell
CL:0000001,primary cultured cell,primary cultured cell
```

## Descriptions

The ontology descriptions and the label descriptions of runs share one JSON format:

```json
{
  "schemaVersion": "3.0",
  "artifactKey": {
    "provider": "nebius",
    "model": "moonshotai/Kimi-K2.6",
    "company": "moonshotai",
    "modelName": "Kimi-K2.6"
  },
  "updatedAt": "<UTC timestamp>",
  "descriptions": {
    "CL:0000815": {
      "initialLabel": "regulatory t cell;regulatory t lymphocyte;...",
      "descriptiveName": "CD4-positive regulatory T lymphocyte",
      "function": "...",
      "diseaseRelevance": "...",
      "developmentalStage": "..."
    }
  }
}
```

The keys of `descriptions` are Cell Ontology IDs in the ontology file and labels in the
files of runs. Each entry has the fields of
[`CellDescription`](api/descriptions.html#cyteonto.CellDescription). Files without
`"schemaVersion": "3.0"`, such as files of older versions, are not loaded. CyteOnto then
generates the descriptions again: for the ontology, all 2,902 of them. Run
`python -m cyteonto.setup --force` instead to download the current files of the
precomputed model pairs.

## Embeddings

The embedding files are compressed NumPy archives:

| Array | Ontology files | Run files |
|---|---|---|
| `embeddings` | One float32 row per term, 2,902 × 4,096 for Qwen3-Embedding-8B | One float32 row per distinct label or part |
| `ontology_ids` | The Cell Ontology ID of each row | |
| `labels` | | The label of each row |
| `metadata` | A dict | A dict |

`metadata` holds `schemaVersion`, a `timestamp`, the `llm` and `embedding` artifact keys,
`num_embeddings`, `embedding_dim` and `reasoning`; run files add `run_id`, `kind` and
`identifier`. The IDs, labels and metadata are object arrays, so load the files with
`allow_pickle=True`:

```python
import numpy as np

data = np.load(path, allow_pickle=True)
embeddings = data["embeddings"]
ids = data["ontology_ids"].tolist()
metadata = data["metadata"].item()
```

A run file without `labels`, written by an older version of CyteOnto, is stale:
CyteOnto does not load it, and `purge_stale()` deletes it.

## Decompositions

How the labels of a run were split:

```json
{
  "schemaVersion": "3.0",
  "artifactKey": {"provider": "nebius", "model": "moonshotai/Kimi-K2.6", "company": "moonshotai", "modelName": "Kimi-K2.6"},
  "updatedAt": "<UTC timestamp>",
  "decompositions": {
    "at2 cell-plasma cell doublet": ["at2 cell", "plasma cell"],
    "t cell": ["t cell"]
  }
}
```

## Downloads

`python -m cyteonto.setup` downloads the files from Nygen's public storage at
`https://pub-d8bf3af01ebe421abded39c4cb33d88a.r2.dev/cyteonto_v2/`, under
`cell_ontology/<file name>`, `descriptions/descriptions_<llmKey>.json` and
`embeddings/embeddings_<llmKey>_<embdKey>.npz`.
