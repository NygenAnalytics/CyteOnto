---
description: How CyteOnto splits labels that name several cell types, such as doublets and mixed populations, scores them part by part, and reduces the part scores with the max or hungarian_mean method.
---

(compound-labels)=
# Compound labels

Some labels name more than one cell type: doublets, such as
`AT2 cell-plasma cell doublet`, or mixed populations, such as `AT2 cell / plasma cell`.
Embedded as one string, such a label matches a single term that fits none of its cell
types well. CyteOnto therefore splits these labels into their cell types, matches each
part to a Cell Ontology term and scores the parts.

## How labels are split

For each distinct label, the LLM decides whether it names one cell type or several and
returns the parts. The prompt sets these rules:

- Text separated by semicolons lists synonyms of one cell type, and is not split.
- Commas and hyphens inside a cell type name do not separate cell types, so
  `CD8-positive, alpha-beta T cell` stays whole.
- Qualifiers such as `doublet`, `mixed population` or `contamination` are removed from
  the parts, and the parts are lowercase.

`AT2 cell-plasma cell doublet`, for example, becomes the parts `at2 cell` and
`plasma cell`. A label is compound only when the model returns two or more parts; if it
returns one, or if the request fails after its retries, the label stays whole. The
model's answers are cached, but a failed request is not, so the next call with the run
tries to split that label again; see [Caching](#compound-labels-caching). With
`use_cache=False`, a failed request keeps the cached split of the label, if there is
one. The parts are
described, embedded and matched like any other label.

## How compound pairs are scored

A pair is compound when its author label or its algorithm label has more than one part.
For *m* author parts and *n* algorithm parts, CyteOnto scores every author part against
every algorithm part with the chosen [metric](similarity-metrics.md), giving an *m* × *n*
matrix *S*. A part that matched no term scores 0. `compound_scoring` reduces *S* to one
score:

- **`max`**, the default, takes the highest entry of *S*.
- **`hungarian_mean`** pairs the parts one to one: it picks the `min(m, n)` pairs with
  the highest total score (the Hungarian algorithm) and takes their mean. When *m* and
  *n* differ, it multiplies the mean by the coverage `min(m, n) / max(m, n)`, so that
  missing or extra cell types lower the score.

The scores below are illustrative part scores, not the output of a run. They show how
the two methods treat the same matrix:

| Author label | Algorithm label | *m* × *n* | *S* | `max` | `hungarian_mean` |
|---|---|---|---|---|---|
| AT2 cell-plasma cell doublet | plasma cell | 2×1 | `[[0.05], [0.92]]` | 0.92 | 0.46 |
| AT2 cell-plasma cell doublet | AT2 cell / plasma cell | 2×2 | `[[0.98, 0.06], [0.05, 0.97]]` | 0.98 | 0.975 |
| AT2 cell-plasma cell doublet | AT2 cell / T cell | 2×2 | `[[0.98, 0.05], [0.06, 0.08]]` | 0.98 | 0.53 |
| AT2 cell-plasma cell doublet | T cell / B cell | 2×2 | `[[0.06, 0.05], [0.08, 0.07]]` | 0.08 | 0.065 |
| AT2 cell / plasma cell / NK cell | AT2 cell / T cell | 3×2 | `[[0.98, 0.05], [0.06, 0.08], [0.05, 0.07]]` | 0.98 | 0.3533 |

Choose the method by the question you ask:

- **`max`** asks whether the two labels share at least one cell type. A doublet scores
  as high against either of its cell types as against itself.
- **`hungarian_mean`** asks whether the two labels name the same set of cell types. A
  doublet scores at most 0.5 against one of its cell types.

```python
df = await cyto.compare(
    author_labels=["AT2 cell-plasma cell doublet", "T cell"],
    algorithms={"method_a": ["Plasma cell", "T cell"]},
    run_id="doublets",
    compound_scoring="hungarian_mean",
)
```

In the hosted API, set the request field `compoundScoring`.

## Compound rows in the results

- `similarity_method` is `cytescore_compound`, also when no part matched a term; the
  score is then 0.
- `author_label` and `algorithm_label` hold the whole labels, lowercased.
- `author_ontology_id`, `author_ontology_name` and `author_embedding_similarity`, and the
  matching algorithm columns, list every part in order, separated by semicolons, such as
  `CL:0002063;CL:0000786`. A part that matched no term leaves an empty entry in the IDs
  and names.

Pairs in which both labels have one part are scored as usual, so a run can mix both
kinds of rows. See [Read the results](results.md).

(compound-labels-caching)=
## Caching

The parts of each label are saved per `run_id`, in
`<user_dir>/decompositions/<run_id>/`, and shared by the author labels and all algorithms
of the run. `clear_run()` does not delete them; delete the folder to split the labels of
the run again, or pass `use_cache=False`. See [Caching and runs](caching.md).
