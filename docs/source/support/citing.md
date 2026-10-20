---
description: How to cite CyteOnto, what to record so that others can reproduce a comparison, and the license of the cyteonto package.
---

(citing)=
# Cite CyteOnto

CyteOnto is part of [CyteType](https://docs.nygen.io/CyteType/). If you use CyteOnto in
your work, please cite the CyteType preprint:

> Ahuja G, Antill A, Su Y, Dall'Olio GM, Basnayake S, Karlsson G, Dhapola P. Multi-agent AI enables evidence-based cell annotation in single-cell transcriptomics. *bioRxiv* 2025. doi:[10.1101/2025.11.06.686964](https://doi.org/10.1101/2025.11.06.686964)

BibTeX:

```bibtex
@article{cytetype2025,
  title   = {Multi-agent {AI} enables evidence-based cell annotation in single-cell transcriptomics},
  author  = {Ahuja, Gautam and Antill, Alex and Su, Yi and Dall'Olio, Giovanni Marco and Basnayake, Sukhitha and Karlsson, G{\"o}ran and Dhapola, Parashar},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.06.686964},
  url     = {https://www.biorxiv.org/content/10.1101/2025.11.06.686964v1}
}
```

(citing-record)=
## Record what you ran

The scores depend on the models and the options of the comparison, and the LLM's
descriptions can differ between runs. Record these items for each comparison you
report:

| Item | Where to find it |
|---|---|
| CyteOnto version | `cyteonto.__version__`, and the commit you installed |
| Model pair | The provider and model of `LlmConfig` and `EmbdConfig`, or of the request to the hosted API |
| Fallback use | `cyto.model_pair_usage.to_status_dict()`, or the `modelPairUsage` of the job's status. When a tier is not `primary`, it names the fallback model; in Python, the fallback LLM only if you passed `fallback_llm` |
| Options | `metric`, `metric_params`, `min_match_similarity` and `compound_scoring`, or the request body without its API keys |
| Run ID | The `run_id` column of the results |
| Results | The result DataFrame or file, with the label pairs, the matched terms and the scores |

Keep the files behind the scores: the run caches in `user_dir`, which hold the
descriptions and embeddings of your labels, and the ontology files of the model pair in
`data_dir`, against which the labels were matched and scored. See
[Caching and runs](../guides/caching.md).

## License

The `cyteonto` package is released under the
[MIT license](https://github.com/NygenAnalytics/CyteOnto/blob/master/LICENSE). The Cell
Ontology is published by the [OBO Foundry](https://obofoundry.org/ontology/cl.html) under
its own license.
