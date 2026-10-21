---
myst:
  html_meta:
    description: Reference for the CyteOnto class, which prepares the Cell Ontology embeddings, compares author labels with algorithm labels and manages the caches of comparison runs.
---

# CyteOnto class

`CyteOnto` compares cell type labels in Cell Ontology space. `from_config()` creates it
and makes sure that the descriptions and embeddings of the Cell Ontology terms are on
disk. `compare()` scores author labels against the labels of one or more algorithms and
returns a DataFrame, and `compare_anndata()` reads the algorithm labels from
`adata.obs`. `cache_stats()`, `clear_run()` and `purge_stale()` manage the caches of
comparison runs.

```python
from cyteonto import CyteOnto

cyto = await CyteOnto.from_config(agent=agent, embedding=embedding, llm=llm)
df = await cyto.compare(
    author_labels=["alveolar macrophage", "regulatory T cell"],
    algorithms={"method_a": ["lung macrophage", "Treg"]},
    run_id="lung_v1",
)
```

`from_config()`, `compare()` and `compare_anndata()` are coroutines: `await` them in a
notebook, or run them with `asyncio.run()` in a script. The
[Python quickstart](../getting-started/quickstart-python) runs these steps,
[Read the results](../guides/results) describes the DataFrame, and
[Caching and runs](../guides/caching) describes what each call saves on disk.

```{eval-rst}
.. autoclass:: cyteonto.CyteOnto
   :members: from_config, compare, compare_anndata, cache_stats, clear_run, purge_stale

   .. py:attribute:: usage
      :type: cyteonto.AgentUsage

      The requests and tokens of the successful LLM calls of this instance, to
      describe and to split labels, including those of ``from_config()``. Failed
      attempts and embedding requests are not counted.

   .. py:attribute:: model_pair_usage
      :type: cyteonto.ModelPairUsage

      Whether the primary or the fallback LLM and embedding models were used.
```
