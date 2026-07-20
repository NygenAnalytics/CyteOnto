# Workflow

CyteOnto compares parallel lists of cell type labels (author reference vs one or more algorithm predictions) by mapping each label into the [Cell Ontology (CL)](https://obofoundry.org/ontology/cl.html) and scoring pairs with an ontology-aware metric. The implementation lives in the `cyteonto` package; see [cyteonto/README.md](../cyteonto/README.md) for the full API reference.

## End-to-end flow

1. **Setup (once per environment)** — `uv run python cyteonto/setup.py` downloads CL assets and precomputed ontology descriptions/embeddings. Then `await CyteOnto.from_config(agent, embedding, llm)` ensures the active model pair is ready (generates missing descriptions/embeddings if needed).
2. **Compare (per analysis)** — `await cyto.compare(author_labels, algorithms={...}, run_id=...)` lowercases labels, decomposes mixture labels when needed, describes and embeds label parts, matches them to CL terms, and scores each author/algorithm pair.
3. **Persist and reuse** — Descriptions (JSON), embeddings (NPZ), and label decompositions (JSON) are written under `user_files/` keyed by `run_id`, so reruns with the same labels skip redundant LLM calls.

## Setup: `setup.py` and `from_config`

**`setup.py`** (CDN download):

1. Required: `cell_to_cell_ontology.csv`, `cl.owl`, primary LLM descriptions JSON, primary embedding NPZ.
2. Optional backup pair artifacts: failures log a warning and setup continues with the primary pair only.
3. Enriched CSV: download `cell_to_cell_ontology_enriched.csv` when available; otherwise build it locally from the original CSV (`label_normalized` = lowercase `label`, drop duplicate `(ontology_id, label_normalized)` rows).

**`from_config`**:

1. Prefer `cell_to_cell_ontology_enriched.csv` for mapping when present; otherwise load the original CSV and normalize labels in memory.
2. Verify `cl.owl` exists under `data_dir`.
3. Load or generate LLM descriptions for every CL term (per text model).
4. Embed those descriptions and save an ontology NPZ (per text + embedding model pair).

If both ontology artifacts already exist and every CL id has a non-blank description, `from_config` returns immediately. Use `force_regenerate=True` to delete and rebuild the ontology cache.

## Compare: `compare`

For each call:

1. **Resolve `run_id`** — Use the value you pass, or an auto-generated `run-<uuid4>` (logged at INFO and stored in every result row).
2. **Lowercase labels** — Non-empty author and algorithm labels are lowercased so casing does not split describe/embed/decompose caches. Result columns echo the lowercased strings.
3. **Decompose labels** — Unique non-empty labels on each side are passed through `decompose_labels`. Mixture labels (doublets, mixed populations) split into lowercase cell-type parts; simple labels map to a single part. Cached under `user_files/decompositions/<run_id>/`.
4. **Describe and embed parts** — Load cached descriptions/embeddings when possible; generate missing ones via LLM; embed unique parts; save under `user_files/.../<run_id>/author/` and per-algorithm paths.
5. **Match to CL** — Cosine similarity between part embeddings and the precomputed ontology embedding matrix. Matches below `min_match_similarity` (default `0.1`) are treated as unmatched (empty ontology id).
6. **Per algorithm** — Repeat decompose/describe/embed/cache/match for each algorithm label list (same length as `author_labels`).
7. **Pair scoring** — For each aligned index:
   - **Simple pair** (one part on each side): `OntologySimilarity.similarity(...)` when both parts matched; otherwise `0.0` with `partial_match` or `no_matches`.
   - **Compound pair** (more than one part on either side): build an m×n cytescore matrix `S`, then reduce with `compound_scoring` (default `"max"` → `max(S)`; `"hungarian_mean"` → Hungarian assignment mean with coverage when `m ≠ n`); `similarity_method = cytescore_compound`.
   - **Empty labels**: both empty → `empty`; one empty → non-empty side keeps ontology fields, score `0.0`, `similarity_method = empty`.
8. **Results** — A `pandas.DataFrame` with one row per `(algorithm, pair_index)`.

Pass `use_cache=False` to skip on-disk lookups and regenerate everything for that call.

### Result columns

| Column | Meaning |
|--------|---------|
| `run_id` | Namespace used for caches and result tagging |
| `algorithm` | Key from the `algorithms` mapping |
| `pair_index` | Index into the parallel label lists (0-based) |
| `author_label`, `algorithm_label` | Input strings after compare-time lowercasing |
| `author_ontology_id`, `algorithm_ontology_id` | Best CL match per part, semicolon-separated in part order. Empty string if unmatched. |
| `author_ontology_name`, `algorithm_ontology_name` | Primary CSV label (or OWL fallback) for each id above |
| `author_embedding_similarity`, `algorithm_embedding_similarity` | Per-part cosine similarity to the matched CL term. Single float when one part; semicolon-separated floats when multiple parts. |
| `cytescore_similarity` | Score from the chosen metric / compound reducer when applicable; else `0.0` |
| `similarity_method` | How the row was classified (see below) |

### `similarity_method` values

| Value | When |
|-------|------|
| `cytescore` | Simple pair; both parts matched valid `CL:` ids; hierarchy/embedding metric applied |
| `cytescore_compound` | Compound pair; score reduced with `compound_scoring` (`max` or `hungarian_mean`) |
| `partial_match` | Exactly one side matched the ontology |
| `no_matches` | Neither side matched |
| `string_similarity` | Both ids present but not standard `CL:` prefixes (rare) |
| `empty` | One or both raw labels are empty |

### Compound scoring (summary)

When either side has more than one part after decomposition:

1. Score every author-part vs algorithm-part pair with the chosen `metric` → matrix `S`.
2. Reduce `S` with `compound_scoring`:
   - **`max` (default):** `cytescore_similarity = max(S)`.
   - **`hungarian_mean`:** select `k = min(m,n)` assignments that maximize total score; mean those `k` scores; if `m ≠ n`, multiply by coverage `min(m,n) / max(m,n)`.
3. Ontology ids, names, and embedding similarities in the result list **all** parts on that side (not only Hungarian-matched pairs).

See `notebooks/quick_tutorial.ipynb` for worked examples.

## Run organization

Comparisons are grouped by **`run_id`**, not by a separate “study” concept:

- Reuse the same `run_id` when you rerun with the same labels to hit the cache.
- Use a new `run_id` for a distinct analysis so caches stay isolated.
- Delete embeddings and descriptions with `cyto.clear_run(run_id)` (optionally scoped to author or a single algorithm). Decomposition JSON under `decompositions/<run_id>/` is separate; remove it manually if needed.

Identifiers are normalized for paths (`/`, `:`, spaces, `.` replaced) — see [FILE_MANAGEMENT.md](FILE_MANAGEMENT.md).

## Similarity metrics

The default `metric="cosine_kernel"` applies a Gaussian hill on raw embedding cosine between the two matched CL term vectors. Other options include `cosine_direct`, OWL hierarchy metrics (`path`, `set:jaccard`, ...), and `simple` (string fallback on CL id strings). See the metrics table in [cyteonto/README.md](../cyteonto/README.md#similarity-metrics).

For simple pairs, `cytescore_similarity` requires both parts to map to CL ids above the match threshold. Compound pairs use the reducer described above.

## Workflow diagram (overview)

```mermaid
flowchart TD
    subgraph setup ["Setup (setup.py + from_config)"]
        H["Load CL CSV / enriched CSV + OWL"]
        I["LLM descriptions for CL terms"]
        J["Embed CL descriptions"]
        K["Save ontology JSON + NPZ"]
        H --> I --> J --> K
    end

    subgraph compare ["Compare (per run_id)"]
        A["Input: author + algorithm labels"]
        L0["Lowercase non-empty labels"]
        D0["LLM decompose mixture labels"]
        B["LLM descriptions for label parts"]
        C["Embed description text"]
        D["Match parts to CL via cosine similarity"]
        E["Score per pair (simple or compound reducer)"]
        F["Results DataFrame"]
        A --> L0 --> D0 --> B --> C --> D --> E --> F
    end

    setup --> compare
    D0 --> D0a["Cache decompositions JSON"]
    B --> B1["Cache descriptions JSON"]
    C --> C1["Cache embeddings NPZ"]
    D --> D1["Ontology embedding matrix"]
```

## Workflow diagram (single label pair)

```mermaid
flowchart TD
    A["Author label + algorithm label"] --> L["Lowercase"]
    L --> B["Decompose each label into parts"]
    B --> C["Describe + embed each unique part"]
    C --> D["Match each part to nearest CL term"]

    D --> E{"Compound pair?"}
    E -->|No| F{"Both parts matched?"}
    F -->|Yes| G["metric e.g. cosine_kernel on CL ids"]
    F -->|No| H["cytescore_similarity = 0.0"]
    G --> I["similarity_method: cytescore"]
    H --> J{"Which side matched?"}
    J -->|Neither| K["no_matches"]
    J -->|One| Lpm["partial_match"]

    E -->|Yes| M{"compound_scoring"}
    M -->|max| Nmax["max of score matrix S"]
    M -->|hungarian_mean| Nhm["Hungarian mean + coverage"]
    Nmax --> N["similarity_method: cytescore_compound"]
    Nhm --> N

    I --> O["Result row"]
    K --> O
    Lpm --> O
    N --> O
```

## AnnData entry point

`compare_anndata` reads algorithm columns from `adata.obs` and delegates to `compare` with the same `run_id`, `compound_scoring`, and caching semantics. Author labels are still passed explicitly as a list and are lowercased the same way.

## Related documentation

- [FILE_MANAGEMENT.md](FILE_MANAGEMENT.md) — Directory layout, naming, and cache utilities
- [cyteonto/README.md](../cyteonto/README.md) — Configuration, metrics, storage formats, and extension points
