# cyteonto

Semantic comparison of cell type annotations against the [Cell Ontology (CL)](https://obofoundry.org/ontology/cl.html).

The documentation at [docs.nygen.io/CyteOnto](https://docs.nygen.io/CyteOnto/) describes how to use the package: the [Python quickstart](https://docs.nygen.io/CyteOnto/getting-started/quickstart-python/), the [API reference](https://docs.nygen.io/CyteOnto/api/index.html), [models and providers](https://docs.nygen.io/CyteOnto/guides/models-and-providers/), [compound labels](https://docs.nygen.io/CyteOnto/guides/compound-labels/), [similarity metrics](https://docs.nygen.io/CyteOnto/guides/similarity-metrics/), [caching](https://docs.nygen.io/CyteOnto/guides/caching/) and the [data files](https://docs.nygen.io/CyteOnto/reference/files/). This file describes the code.

---

## Package layout

```
cyteonto/
├── __init__.py       Public exports
├── config.py         Model defaults, provider URLs and API key variables, limits, result columns
├── logger.py         Loguru configuration
├── paths.py          PathConfig, which resolves the file locations
├── models.py         CellDescription, LabelDecomposition, EmbdConfig, LlmConfig, ModelArtifactKey, ModelPairUsage, AgentUsage
├── storage.py        NPZ and JSON read/write (ontology + user variants)
├── embed.py          Async HTTP embedding generation
├── describe.py       LLM description and compound-label decomposition
├── ontology.py       OntologyMapping (CSV) and OntologySimilarity (OWL + metrics)
├── cyteonto.py       CyteOnto orchestrator class
├── setup.py          Downloads the Cell Ontology files and precomputed artifacts (python -m cyteonto.setup)
└── data/             Downloaded and generated data (not tracked)
```

### Module dependency graph

```
                  config
                    |
                  logger
                    |
  models -- paths --+-- storage
                    |
          embed ----+---- describe
                    |
               ontology
                    |
                cyteonto
                    |
                 __init__
```

Lower layers do not import higher ones. The `CyteOnto` class in `cyteonto.py` is the only module that wires everything together.

### Call graphs

```
CyteOnto.from_config
 └─ _prepare_ontology
     ├─ storage.load_descriptions                  ontology descriptions of the LLM
     ├─ _describe_labels(terms without one)        describe.describe_cells, then the fallback agent
     ├─ storage.save_descriptions                  after deleting the NPZs of the old descriptions
     └─ _embed_ontology                            embed.embed_texts, save NPZ
         └─ _load_or_embed_ontology                if the primary fails: check the fallback, its NPZ, then switch

CyteOnto.compare
 ├─ _resolve_label_parts(author_labels)            LLM decompose + decompositions JSON cache, shared by the sides of the call
 ├─ _embed_user_labels(author parts)
 │   ├─ storage.load_descriptions / load_user_embeddings
 │   ├─ _describe_labels(missing or blank)
 │   ├─ storage.save_descriptions                  after deleting the NPZs of the old descriptions
 │   ├─ _embed_with_failover(texts)                embed.embed_texts, then the fallback embedding
 │   │   └─ _load_or_embed_ontology                at a failover: NPZ of the fallback model, then switch
 │   ├─ _record_embedding_tier                     model whose label embeddings are matched
 │   └─ storage.save_user_embeddings
 ├─ _match(author part embeddings)
 ├─ _ensure_similarity                             OntologySimilarity (OWL + ontology embeddings)
 └─ for each algorithm:
     ├─ _resolve_label_parts(algo_labels)
     ├─ _embed_user_labels(algo parts)
     ├─ _match(algo part embeddings)
     ├─ _ensure_similarity                         reloaded after an embedding failover
     └─ for each pair_index:
         ├─ simple: OntologySimilarity.similarity(a_id, g_id, ...)
         └─ compound: build S; max(S) or _hungarian_match_mean(S)
```

---

## Extending

- **New embedding provider**: add its endpoint to `Config._PROVIDER_URL` and its API key variable to `Config.PROVIDER_API_KEY_ENV` in `config.py`, extend the `EmbdProvider` `Literal` in `models.py` (and in `modal_app/models.py` to accept it in the HTTP API), and adjust `_headers` / `_build_payload` / `_extract_embedding` in `embed.py` if the request or response shape differs. Add the provider to the table in `docs/source/guides/models-and-providers.md`; the tests compare `_PROVIDER_URL` with `EmbdProvider` and with that table.
- **New similarity metric**: implement a helper in `OntologySimilarity`, add a branch in `OntologySimilarity.similarity`, and add the metric to the list in the `OntologySimilarity` docstring, to the `compare` docstring and to `docs/source/guides/similarity-metrics.md`.
- **Alternative prompt**: edit `describe._build_prompt` for descriptions or `describe._build_decompose_prompt` for decompositions. The description prompt lists every `CellDescription` field with a soft length cap and a neutral tone, and embeds no examples to avoid anchoring; keep the same structure. Descriptions and decompositions are cached per model, not per prompt, so delete the cached ones after changing a prompt.
- **Different ontology files**: pass a custom `data_dir` with `cell_ontology/cl.owl`, `cell_ontology/cell_to_cell_ontology.csv` and `cell_ontology/cell_to_cell_ontology_enriched.csv`. The enriched CSV adds a lowercase `label_normalized` column and drops duplicate `(ontology_id, label_normalized)` rows; `main()` in `setup.py` shows how it is built.
