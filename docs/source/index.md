---
description: CyteOnto compares cell type labels in Cell Ontology space. It matches the labels of a study's authors and of annotation algorithms to Cell Ontology terms and scores each pair, from a hosted HTTP API or the cyteonto Python package.
---

(home)=
# CyteOnto

CyteOnto compares cell type annotations. You give it the reference labels of a study,
the author labels, and the labels that one or more annotation algorithms assigned to the
same cells or clusters. CyteOnto matches every label to a term of the
[Cell Ontology](https://obofoundry.org/ontology/cl.html) (CL) and scores how similar
the author's term and the algorithm's term are. Labels that match the same term, as
`Treg` and `regulatory T cell` usually do, score 1. `CD4-positive, alpha-beta T cell`
scores close to `CD8-positive, alpha-beta T cell`, and `T cell` scores low against
`neuron`. Use it to benchmark annotation methods against a reference without string
matching.

## How it works in brief

1. An LLM describes each label, after splitting mixture labels such as doublets into
   their cell types.
2. An embedding model turns each description into a vector, which is matched to the
   closest of the 2,902 Cell Ontology terms.
3. The two terms that an author label and an algorithm label matched are scored with a
   similarity metric, which compares the embeddings of the terms' descriptions by
   default and can also use the ontology's hierarchy.

[How CyteOnto works](getting-started/how-cyteonto-works.md) describes each step.

## Two ways to use CyteOnto

|  | Hosted API | Python package |
|---|---|---|
| Runs on | `https://cyteonto.nygen.io` | Your computer |
| You need | An HTTP client | Python 3.13 or newer |
| API keys | None with the default models | Your own, for an LLM and an embedding provider |
| Options | The fields of the request | Every option, plus the caches and files on disk |

## A first comparison

This request compares three author labels with the labels of one algorithm on the hosted
API. It needs no API key:

```bash
curl -sS -X POST https://cyteonto.nygen.io/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "authorLabels": ["alveolar macrophage", "regulatory T cell", "CD8-positive, alpha-beta T cell"],
    "algorithms": {"method_a": ["lung macrophage", "Treg", "CD8 T cell"]}
  }'
```

The response holds a run ID. The job runs in the background: poll its status, then
download one row per label pair. The
[hosted API quickstart](getting-started/quickstart-api.md) walks through these steps
with curl and Python, and the [Python quickstart](getting-started/quickstart-python.md)
runs the same comparison with the `cyteonto` package.

## Where to go next

- **Getting started:** [install the package](getting-started/installation.md), compare
  labels with the [hosted API](getting-started/quickstart-api.md) or
  [in Python](getting-started/quickstart-python.md), and read
  [how CyteOnto works](getting-started/how-cyteonto-works.md).
- **Guides:** choose [models and providers](guides/models-and-providers.md), compare
  [compound labels](guides/compound-labels.md), pick a
  [similarity metric](guides/similarity-metrics.md), reuse and clean
  [caches](guides/caching.md), compare [labels in AnnData](guides/anndata.md), and
  [read the results](guides/results.md).
- **Reference:** the [Python API](reference/api/index.html), the
  [HTTP API](reference/http-api.md), the
  [environment variables and limits](reference/configuration.md) and the
  [data files and formats](reference/files.md).
- **Support:** [troubleshooting](support/troubleshooting.md) and
  [how to cite CyteOnto](support/citing.md).

## Source code and issues

CyteOnto is developed in
[NygenAnalytics/CyteOnto](https://github.com/NygenAnalytics/CyteOnto) under the MIT
license. Report problems in the
[CyteOnto issues](https://github.com/NygenAnalytics/CyteOnto/issues). CyteOnto is part of
[CyteType](https://docs.nygen.io/CyteType/), which annotates the clusters of
single-cell RNA-seq data. The
[documentation index for LLMs](https://docs.nygen.io/CyteOnto/llms.txt) lists every page
in plain text for assistants that read this site.
