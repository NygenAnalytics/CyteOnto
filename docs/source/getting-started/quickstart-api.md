---
description: Compare cell type labels with the hosted CyteOnto API at cyteonto.nygen.io, with curl or Python, without installing anything or bringing API keys.
---

(quickstart-api)=
# Quickstart (hosted API)

The hosted API runs CyteOnto for you at `https://cyteonto.nygen.io`. With the default
models it needs no API key. A comparison is a job: you submit the labels, poll the job's
status, and download the results when it has finished. This page compares three author
labels with the labels of two algorithms.

The curl examples use [jq](https://jqlang.org/) to read the JSON responses.

## Submit the labels

Send the author labels and, for each algorithm, a list of labels of the same length.
The labels at the same position are compared with each other.

::::{tab-set}
:::{tab-item} curl
:sync: curl
```bash
export CYTEONTO_URL="https://cyteonto.nygen.io"

RUN_ID=$(curl -sS -X POST "$CYTEONTO_URL/compare" \
  -H 'Content-Type: application/json' \
  -d '{
    "authorLabels": ["alveolar macrophage", "regulatory T cell", "CD8-positive, alpha-beta T cell"],
    "algorithms": {
      "method_a": ["lung macrophage", "Treg", "CD8 T cell"],
      "method_b": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"]
    }
  }' | jq -r .runId)
echo "$RUN_ID"
```
:::
:::{tab-item} Python
:sync: python
```python
import time

import pandas as pd
import requests

BASE_URL = "https://cyteonto.nygen.io"

# Submit the author labels and the labels of two algorithms.
response = requests.post(
    f"{BASE_URL}/compare",
    json={
        "authorLabels": [
            "alveolar macrophage",
            "regulatory T cell",
            "CD8-positive, alpha-beta T cell",
        ],
        "algorithms": {
            "method_a": ["lung macrophage", "Treg", "CD8 T cell"],
            "method_b": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"],
        },
    },
    timeout=30,
)
response.raise_for_status()
run_id = response.json()["runId"]
print(run_id)
```
:::
::::

The response holds the run ID of the new job, in the form `run-<uuid>`, and its state:

```json
{"runId": "run-<uuid>", "state": "queued"}
```

Keep the run ID: you need it for every other request about the job.

## Wait for the job

Poll `/status/<run ID>` until the state is `completed` or `failed`. A job normally
moves from `queued` to `running` and ends in one of these two states, but a job that
stops unexpectedly can keep its last state, so stop polling after a time limit.

::::{tab-set}
:::{tab-item} curl
:sync: curl
```bash
# Poll every 10 seconds, for at most an hour.
for _ in $(seq 360); do
  STATE=$(curl -sS "$CYTEONTO_URL/status/$RUN_ID" | jq -r .state)
  echo "$STATE"
  if [ "$STATE" = completed ] || [ "$STATE" = failed ]; then break; fi
  sleep 10
done
```
:::
:::{tab-item} Python
:sync: python
```python
# Poll the status every 10 seconds until the job has finished, for at most an hour.
deadline = time.time() + 3600
while True:
    status = requests.get(f"{BASE_URL}/status/{run_id}", timeout=30).json()
    if status["state"] in ("completed", "failed"):
        break
    if time.time() > deadline:
        raise TimeoutError(f"{run_id} is still {status['state']} after an hour")
    time.sleep(10)

# A failed job reports the exception that stopped it.
if status["state"] == "failed":
    raise RuntimeError(status["error"])
```
:::
::::

The status also counts the labels and the result rows, and records which models
served the job:

```json
{
  "runId": "run-<uuid>",
  "state": "completed",
  "createdAt": "<UTC timestamp>",
  "startedAt": "<UTC timestamp>",
  "completedAt": "<UTC timestamp>",
  "error": null,
  "numAuthorLabels": 3,
  "numAlgorithms": 2,
  "numRows": 6,
  "resultCsvPath": "user_files/run-<uuid>/result.csv",
  "resultJsonPath": "user_files/run-<uuid>/result.json",
  "modelPairUsage": {
    "llmTier": "primary",
    "embeddingTier": "primary",
    "llmProvider": "nebius",
    "llmModel": "moonshotai/Kimi-K2.6",
    "embeddingProvider": "nebius",
    "embeddingModel": "Qwen/Qwen3-Embedding-8B"
  }
}
```

## Download the results

Fetch the results as JSON, a list with one object per row, or as a CSV file:

::::{tab-set}
:::{tab-item} curl
:sync: curl
```bash
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=json" | jq
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=csv" -o "$RUN_ID.csv"
```
:::
:::{tab-item} Python
:sync: python
```python
# Download the rows as JSON and load them into a DataFrame.
rows = requests.get(
    f"{BASE_URL}/result/{run_id}", params={"format": "json"}, timeout=60
).json()
df = pd.DataFrame(rows)

# Show each label pair, the matched Cell Ontology terms and the score.
df[
    [
        "algorithm",
        "author_label",
        "algorithm_label",
        "author_ontology_name",
        "algorithm_ontology_name",
        "cytescore_similarity",
    ]
]
```
:::
::::

There is one row for each algorithm and position, six here. The main columns are:

| Column | Meaning |
|---|---|
| `algorithm`, `pair_index` | The algorithm and the position of the label pair, from 0 |
| `author_label`, `algorithm_label` | The two labels, lowercased |
| `author_ontology_id`, `algorithm_ontology_id` | The Cell Ontology term that each label matched, such as `CL:0000815` |
| `author_ontology_name`, `algorithm_ontology_name` | The names of those terms |
| `cytescore_similarity` | The similarity of the two terms: 1 for the same term, and lower the less related they are |
| `similarity_method` | How the row was scored, such as `cytescore`, or why it was not, such as `partial_match` |

[Read the results](../guides/results.md) explains every column and how to interpret the
scores.

## Next steps

- Choose the metric, the match threshold and other options with the fields of the
  request: [HTTP API](../reference/http-api.md).
- Labels such as `AT2 cell-plasma cell doublet` name several cell types; see
  [Compound labels](../guides/compound-labels.md).
- To run CyteOnto yourself, with your own models and caches, see
  [Quickstart (Python)](quickstart-python.md).

:::{note}
The service has no accounts: anyone who has a run ID can read the job's status and
results. Run IDs are random, but share them only with people who may see your labels.
The service sends your labels to its LLM and embedding providers, and keeps the results,
descriptions and embeddings of each job.
:::
