---
description: The similarity metrics that CyteOnto uses to score two Cell Ontology terms, from the default cosine_kernel to the hierarchy metrics, with reference scores, the kernel parameters and the match threshold.
---

(similarity-metrics)=
# Similarity metrics

The score of a pair compares the two Cell Ontology terms that the labels matched, not the
labels themselves. Two labels that matched the same term score 1 under every metric.
For different terms, `metric` decides how they are compared: by the embeddings of their
descriptions, by their places in the ontology's hierarchy, or by both.

```python
df = await cyto.compare(
    author_labels=author_labels,
    algorithms=algorithms,
    run_id="lung_v1_path",
    metric="path",
)
```

In the hosted API, set the request fields `metric` and `metricParams`.

## Metrics

| `metric` | Uses | Score |
|---|---|---|
| `cosine_kernel` (default) | Embeddings | A Gaussian of the cosine similarity of the two terms' description embeddings, which is 1 at a cosine of 1 |
| `cosine_direct` | Embeddings | The cosine similarity of the two terms' description embeddings |
| `path` | Hierarchy | 1 over the mean depth of the two terms below their deepest common ancestor. Sibling terms score 1, and a term and its child score less |
| `set:jaccard` | Hierarchy | The Jaccard index of the two sets of ancestors, each including its term |
| `set:cosine` | Hierarchy | The shared ancestors over the square root of the product of the two set sizes, with the terms included |
| `set:weighted_jaccard` | Hierarchy | The Jaccard index of the ancestor sets, with each ancestor weighted by its depth |
| `weighted:num_ancestors` | Hierarchy | The Jaccard index of the ancestors, without the terms, with each ancestor weighted by 1 over its own number of ancestors plus 1 |
| `weighted:specificity` | Hierarchy | The Jaccard index of the ancestors, with each ancestor weighted by its depth relative to the term's depth |
| `weighted:embedding_cosine` | Both | The Jaccard index of the ancestors, with each ancestor weighted by the cosine similarity of its embedding with the term's |
| `simple` | Neither | The character similarity of the two CL IDs, such as `CL:0000084` and `CL:0000236`. It says nothing about the cell types and is only a fallback |

The embedding metrics compare the terms' descriptions, as written by the LLM of the model
pair. The hierarchy metrics use `cl.owl` and depend only on the ontology. Depth is the
longest chain of `is_a` parents from a term up to `cell` (`CL:0000000`). The
`cosine_kernel` and `cosine_direct` scores depend on the model pair, so compare them only
between runs with the same models.

## Reference scores

These scores of some term pairs come from the precomputed embeddings of the default model
pair, Kimi K2.6 with Qwen3-Embedding-8B, and from `cl.owl`:

| Terms | `cosine_kernel` | `cosine_direct` | `weighted:embedding_cosine` |
|---|---|---|---|
| CD4-positive, alpha-beta T cell vs CD8-positive, alpha-beta T cell | 0.77 | 0.82 | 0.97 |
| T cell vs regulatory T cell | 0.64 | 0.76 | 0.66 |
| monocyte vs macrophage | 0.60 | 0.75 | 0.77 |
| macrophage vs alveolar macrophage | 0.46 | 0.69 | 0.69 |
| T cell vs B cell | 0.36 | 0.64 | 0.80 |
| T cell vs neuron | 0.15 | 0.51 | 0.12 |
| hepatocyte vs T cell | 0.13 | 0.49 | 0.14 |

| Terms | `path` | `set:jaccard` | `set:cosine` | `set:weighted_jaccard` | `weighted:num_ancestors` | `weighted:specificity` |
|---|---|---|---|---|---|---|
| CD4-positive, alpha-beta T cell vs CD8-positive, alpha-beta T cell | 1.00 | 0.87 | 0.93 | 0.71 | 1.00 | 1.00 |
| T cell vs regulatory T cell | 0.50 | 0.83 | 0.91 | 0.61 | 0.95 | 0.47 |
| monocyte vs macrophage | 1.00 | 0.74 | 0.85 | 0.62 | 0.85 | 0.81 |
| macrophage vs alveolar macrophage | 0.50 | 0.84 | 0.92 | 0.64 | 0.97 | 0.50 |
| T cell vs B cell | 0.67 | 0.75 | 0.86 | 0.48 | 0.97 | 0.65 |
| T cell vs neuron | 0.33 | 0.13 | 0.24 | 0.03 | 0.29 | 0.04 |
| hepatocyte vs T cell | 0.33 | 0.14 | 0.26 | 0.04 | 0.31 | 0.04 |

The metrics answer different questions:

- `cosine_direct` varies little: across all 4.2 million pairs of the 2,902 terms, half
  of the cosine similarities lie between 0.38 and 0.46, and 95 % are below 0.56.
  `cosine_kernel` spreads them out: 76 % of all pairs score below 0.1, and fewer than
  1 % score 0.5 or more, so related terms stand out from unrelated ones.
- The embedding metrics follow what the descriptions say. A broad term and its specific
  child, such as macrophage and alveolar macrophage, can score lower than two sibling
  terms.
- The hierarchy metrics follow the ontology's structure. `path` gives siblings the
  score of identical terms, and `weighted:num_ancestors` scores T cell and B cell almost
  as high as identical terms.

## Tune `cosine_kernel`

`cosine_kernel` maps the cosine similarity `c` of the two terms' embeddings to a score
with a Gaussian:

```python
score = amplitude * exp(-((c - center) ** 2) / (2 * width**2))
```

`center` is 1.0 by default, `width` 0.25 and `amplitude` 1.0. Pass them in
`metric_params`; other metrics ignore it:

```python
df = await cyto.compare(
    author_labels=author_labels,
    algorithms=algorithms,
    run_id="lung_v1_wide",
    metric="cosine_kernel",
    metric_params={"width": 0.5},
)
```

The width sets how fast the score falls as the terms' cosine similarity drops:

| Cosine similarity | `width=0.1` | `width=0.25` (default) | `width=0.5` |
|---|---|---|---|
| 0.95 | 0.882 | 0.980 | 0.995 |
| 0.90 | 0.607 | 0.923 | 0.980 |
| 0.80 | 0.135 | 0.726 | 0.923 |
| 0.70 | 0.011 | 0.487 | 0.835 |
| 0.60 | 0.000 | 0.278 | 0.726 |
| 0.50 | 0.000 | 0.135 | 0.607 |

A narrow width rewards only near-identical terms; a wide one gives partial credit to more
distant terms. Keep `center` and `amplitude` at 1: identical terms always score 1,
whatever the amplitude.

(similarity-threshold)=
## The match threshold

`min_match_similarity` is the lowest cosine similarity between the embedding of a label's
description and the embedding of its closest term at which the label counts as matched.
It is 0.1 by default, which is low enough for nearly every label to match some term.
Raise it to leave weak matches unmatched. A pair of single-part labels with an unmatched
label scores 0, and its `similarity_method` is `partial_match` or `no_matches`. In a
[compound pair](compound-labels.md), an unmatched part scores 0 against every part of
the other label.

```python
df = await cyto.compare(
    author_labels=author_labels,
    algorithms=algorithms,
    run_id="lung_v1_strict",
    min_match_similarity=0.5,
)
```

To choose a threshold, look at the `author_embedding_similarity` and
`algorithm_embedding_similarity` columns of a run, which hold the similarity of each label
with its closest term, whether or not it was matched.

## Unknown metrics and missing terms

A metric name that CyteOnto does not know falls back to `simple`, and the error is
logged for each pair. The hierarchy metrics also fall back to `simple` for a term that
is not in `cl.owl`. If the scores look wrong, check the log for these messages.
