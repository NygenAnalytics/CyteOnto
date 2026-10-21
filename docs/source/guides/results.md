---
description: The columns of the CyteOnto results, the similarity_method values, how to interpret the CyteScore similarity, and pandas recipes to summarize, check and save a comparison.
---

(results)=
# Read the results

`compare()` returns a pandas DataFrame, and the hosted API returns the same table as
JSON or CSV. It has one row for each algorithm and position: with 100 author labels and
two algorithms, 200 rows, ordered by algorithm and then by position.

## Columns

| Column | Type | Meaning |
|---|---|---|
| `run_id` | str | The run ID of the call |
| `algorithm` | str | The name of the algorithm |
| `pair_index` | int | The position in the label lists, from 0 |
| `author_label` | str | The author label, lowercased |
| `algorithm_label` | str | The algorithm label, lowercased |
| `author_ontology_id` | str | The ID of the Cell Ontology term that the author label matched, such as `CL:0000815`, or an empty string if it matched none |
| `author_ontology_name` | str | The name of that term, such as `regulatory T cell` |
| `author_embedding_similarity` | float or str | The cosine similarity of the label's description with the description of its closest term, also when the label stayed unmatched |
| `algorithm_ontology_id` | str | As `author_ontology_id`, for the algorithm label |
| `algorithm_ontology_name` | str | As `author_ontology_name`, for the algorithm label |
| `algorithm_embedding_similarity` | float or str | As `author_embedding_similarity`, for the algorithm label |
| `cytescore_similarity` | float | The similarity of the two terms under the chosen [metric](similarity-metrics.md), or 0 when a label is empty or unmatched |
| `similarity_method` | str | How the row was scored, or why it was not |

Similarities are rounded to 4 decimals. For a [compound label](compound-labels.md), the
ID, name and embedding similarity columns of its side list every part, separated by
semicolons, and the embedding similarity is then a string, such as `0.7012;0.6634`.
The name of a term is its first label in the ontology mapping file. For nearly all terms
that is the term's name in the Cell Ontology; a few differ, such as `CL:0000649`, which
the mapping names `prickle cell`.

## `similarity_method`

| Value | Meaning |
|---|---|
| `cytescore` | Both labels matched a term and were scored with the metric |
| `cytescore_compound` | At least one label has several parts; the part scores were reduced with `compound_scoring` |
| `partial_match` | Only one of the labels matched a term; the score is 0 |
| `no_matches` | Neither label matched a term; the score is 0 |
| `empty` | One or both labels are empty; the score is 0, and a non-empty label keeps its match |
| `string_similarity` | At least one matched term has an ID that does not start with `CL:`; the score still comes from the metric. The shipped ontology files hold only `CL:` IDs, so this does not occur with them |

A label stays unmatched when the cosine similarity with its closest term is below
`min_match_similarity`; see [The match threshold](similarity-metrics.md#similarity-threshold).

## Interpret the scores

- **1** means, with the default metric, that both labels matched the same Cell Ontology
  term, whatever their wording. Some hierarchy metrics also give 1 to different terms,
  such as sibling terms under `path`, and with `compound_scoring="max"` a compound pair
  scores 1 when a part of each label matched the same term.
- **Below 1**, the score says how related the two terms are. With the default metric
  and model pair, CD4-positive and CD8-positive alpha-beta T cells score 0.77, T cell
  and regulatory T cell 0.64, T cell and B cell 0.36, and T cell and neuron 0.15. Of all
  pairs of terms, 76 % score below 0.1. [Similarity metrics](similarity-metrics.md)
  lists more reference scores.
- **A broad and a specific label**, such as `macrophage` and `alveolar macrophage`, can
  score lower than two related specific types: the default metric compares the terms'
  descriptions, not their places in the hierarchy. Use a hierarchy metric if that
  matters for your question.
- **Check the matches**, not only the scores. The ontology names show what each label
  was taken to mean, and a low embedding similarity shows a label that fits its term
  poorly.

Scores depend on the metric and the model pair, so compare algorithms only within one
set of settings.

## Summaries with pandas

```python
import pandas as pd

# Mean score of each algorithm.
df.groupby("algorithm")["cytescore_similarity"].mean()

# Share of label pairs whose labels matched the same Cell Ontology term.
matched = df["author_ontology_id"] != ""
same_term = matched & (df["author_ontology_id"] == df["algorithm_ontology_id"])
same_term.groupby(df["algorithm"]).mean()

# Rows that were not scored.
df[df["similarity_method"].isin(["partial_match", "no_matches", "empty"])]

# The author labels sorted by how well they fit their term; compound labels last.
fit = pd.to_numeric(df["author_embedding_similarity"], errors="coerce")
df.assign(fit=fit).drop_duplicates("author_label").sort_values("fit")[
    ["author_label", "author_ontology_name", "fit"]
]
```

## Save the results

```python
df.to_csv("cyteonto_results.csv", index=False)
```

When you load the CSV with pandas, pass `keep_default_na=False`, or empty IDs and names
become `NaN`:

```python
df = pd.read_csv("cyteonto_results.csv", keep_default_na=False)
```
