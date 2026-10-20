---
description: How CyteOnto turns cell type labels into Cell Ontology terms and scores them, from the precomputed ontology embeddings to label decomposition, description, embedding, matching and scoring, and what is sent to the model providers.
---

(how-cyteonto-works)=
# How CyteOnto works

CyteOnto does not compare label strings. It maps each label to a term of the Cell
Ontology and compares the terms. Two labels that map to the same term, such as
`Treg` and `regulatory T cell`, score 1 however they are worded, and labels that map to
different terms score by how related those terms are.

```{mermaid}
flowchart TD
    subgraph setup ["Once per model pair"]
        T["Cell Ontology terms and their labels"] --> TD["LLM: describe each term"]
        TD --> TE["Embed each description"]
    end
    subgraph run ["compare()"]
        A["Author and algorithm labels"] --> L["Lowercase"]
        L --> D["LLM: split compound labels into cell types"]
        D --> B["LLM: describe each label or part"]
        B --> E["Embed each description"]
        E --> M["Match to the closest term embedding"]
        M --> S["Score the two terms of each pair"]
        S --> R["One row per algorithm and position"]
    end
    TE --> M
```

## Prepare the ontology

The mapping file lists 6,155 labels, the names and synonyms of 2,902 Cell Ontology
terms. For each term, CyteOnto joins its lowercase labels with semicolons, such as
`alveolar macrophage;dust cell;mf.lu` for `CL:0000583`. The LLM describes each term as a
[`CellDescription`](../reference/api/descriptions.html#cyteonto.CellDescription) with five
fields. This is the description of `CL:0000815`, regulatory T cell, in the precomputed
files:

| Field | Value |
|---|---|
| `initialLabel` | `regulatory t cell;regulatory t lymphocyte;regulatory t-cell;regulatory t-lymphocyte;suppressor t cell;suppressor t lymphocyte;suppressor t-cell;suppressor t-lymphocyte;treg` |
| `descriptiveName` | CD4-positive regulatory T lymphocyte |
| `function` | Suppresses immune responses and maintains self-tolerance by inhibiting effector T cell proliferation and cytokine production. Limits inflammation and prevents autoimmunity through contact-dependent and cytokine-mediated mechanisms. |
| `diseaseRelevance` | Deficiency causes severe autoimmune diseases such as IPEX syndrome. Expanded populations occur in tumors and chronic infections, promoting immune evasion. Adoptive Treg therapy is explored for autoimmune diseases and transplantation. |
| `developmentalStage` | Terminally differentiated; arises in thymus from CD4 single-positive thymocytes or periphery from conventional CD4 T cells |

The embedding model then embeds each description, joined into one sentence. The
descriptions depend on the LLM and the embeddings on both models, so CyteOnto keeps them
in files named after the model pair. `python -m cyteonto.setup` downloads the files of
the default model pair, and `CyteOnto.from_config()` generates any that are missing
(see [Models and providers](../guides/models-and-providers.md)).

## Compare labels

For each call of `compare()`:

1. **Lowercase.** Non-empty labels are lowercased, so that `Plasma cell` and
   `plasma cell` share their caches. The results show the lowercased labels.
2. **Split compound labels.** The LLM decides whether each distinct label names one cell
   type or several, as a doublet or a mixed population does, and splits the latter into
   their cell types. Semicolons separate synonyms of one cell type, and the prompt tells
   the model not to split them. See [Compound labels](../guides/compound-labels.md).
3. **Describe.** The LLM writes a `CellDescription` of each distinct label or part, in
   the same way as for the ontology terms. If a label cannot be described after all
   retries, the label text itself is embedded instead.
4. **Embed.** The embedding model embeds each description.
5. **Match.** CyteOnto computes the cosine similarity of each embedding with the
   embeddings of all 2,902 terms and keeps the closest term. A label whose closest term
   has a similarity below `min_match_similarity`, 0.1 by default, stays unmatched.
6. **Score.** For each position, the author's term and the algorithm's term are scored
   with the chosen metric, by default `cosine_kernel`, which compares the embeddings of
   the two terms' descriptions. Compound labels are scored part by part, and the part
   scores are reduced to one score. A pair with an empty label, or with a single-part
   label that matched no term, scores 0. See
   [Similarity metrics](../guides/similarity-metrics.md).
7. **Return the results.** A DataFrame with one row per algorithm and position. See
   [Read the results](../guides/results.md).

The decompositions, descriptions and embeddings of the labels are saved under the
`run_id` of the call, so a second call with the same labels and `run_id` calls no model
(see [Caching and runs](../guides/caching.md)).

## What CyteOnto sends to the providers

- **To the LLM provider:** each distinct label, in the prompts that split it and
  describe it. Only the model of your pydantic-ai agent is used; CyteOnto sends its own
  instructions.
- **To NCBI:** with `use_pubmed_tool=True`, the default, the model may search PubMed
  while it describes a label, and its search terms are sent to NCBI's E-utilities.
- **To the embedding provider:** the text of each description, which starts with the
  label.

When you run the package yourself, your labels go only to these providers, and to the
fallback providers if you configure them, and the caches stay on your disk.
`python -m cyteonto.setup` only downloads files. With the
[hosted API](quickstart-api.md), your labels go to the CyteOnto service, which calls its
own providers and keeps the results, descriptions and embeddings of each job.

## What it costs

| Step | Model requests |
|---|---|
| `from_config()` with the precomputed files of the model pair | None |
| `from_config()` with another LLM | One description per term, 2,902, and one embedding per term, once |
| `from_config()` with the default LLM and another embedding model | One embedding per term, 2,902, once |
| `compare()` | One request to split each distinct label of the run. Then, separately for the author labels and for each algorithm, one description and one embedding of each distinct label or part |
| `compare()` at the first embedding failover | One embedding per term, 2,902, with the fallback model, unless its ontology embeddings exist |
| `compare()` again with the same `run_id` and labels | None |

So a label that both the authors and an algorithm use is described and embedded twice.
A description takes one more request for each PubMed search, and failed requests are
retried. `CyteOnto.usage` counts the requests and tokens of the LLM calls that
succeeded.
