# CyteOnto: Semantic Cell Type Annotation Comparison Using Large Language Models and Cell Ontology
 
`cyteonto` compares two sets of cell type annotations against the [Cell Ontology (CL)](https://obofoundry.org/ontology/cl.html). Given label lists from a study author and one or more annotation algorithms, it:

1. Decomposes mixture labels (doublets, mixed populations) into cell-type parts with an LLM when needed.
2. Generates a structured description for every label (or part) with an LLM.
3. Embeds those descriptions with a configured embedding model.
4. Matches each embedding to the closest CL term.
5. Scores each author/algorithm pair using an ontology-aware similarity metric (default: a Gaussian kernel on the cosine similarity of the CL term embeddings). Compound pairs default to the maximum entry in the part-by-part score matrix (`compound_scoring="max"`); set `compound_scoring="hungarian_mean"` for Hungarian assignment mean with a coverage penalty when part counts differ.
6. Returns a tidy DataFrame with one row per `(algorithm, pair_index)`.
 
Updated ReadMe: [cyteonto/README.md](cyteonto/README.md). Process flow and file layout: [docs/WORKFLOW.md](docs/WORKFLOW.md), [docs/FILE_MANAGEMENT.md](docs/FILE_MANAGEMENT.md).


## Modal Service

The CyteOnto is available as a service! **No API keys required, we provide it for you!**

### Quick start:

Send a compare request:

```bash
export CYTEONTO_URL="https://cyteonto.nygen.io"

curl -sS -X POST "$CYTEONTO_URL/compare" \
  -H 'Content-Type: application/json' \
  -d '{
    "authorLabels": ["alveolar macrophage", "regulatory T cell", "CD8-positive, alpha-beta T cell"],
    "algorithms": {
      "algo1": ["lung macrophage", "Treg", "CD8 T cell"],
      "algo2": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"]
    }
  }'
```

Response:

```json
{ "runId": "run-<uuid>", "state": "queued" }
```

Save the `runId` and poll status:

```bash
RUN_ID="run-<uuid>"
curl -sS "$CYTEONTO_URL/status/$RUN_ID" | jq
```

Fetch the result once `state == "completed"`:

```bash
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=json" | jq
curl -sS "$CYTEONTO_URL/result/$RUN_ID?format=csv" -o "$RUN_ID.csv"
```


Read the [modal_app/README.md](modal_app/README.md) for more details. Check out the [modal_app/example.py](modal_app/example.py) file to run via Python client.


## Citation

CyteOnto is a part of the [CyteType](https://www.nygen.io/products/cytetype). If you use CyteOnto in your research, please cite our preprint:

> Ahuja G, Antill A, Su Y, Dall'Olio GM, Basnayake S, Karlsson G, Dhapola P. Multi-agent AI enables evidence-based cell annotation in single-cell transcriptomics. *bioRxiv* 2025. doi: [10.1101/2025.11.06.686964](https://www.biorxiv.org/content/10.1101/2025.11.06.686964v1)

```bibtex
@article{cytetype2025,
  title={Multi-agent AI enables evidence-based cell annotation in single-cell transcriptomics},
  author={Gautam Ahuja, Alex Antill, Yi Su, Giovanni Marco Dall'Olio, Sukhitha Basnayake, Göran Karlsson, Parashar Dhapola},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.06.686964},
  url={https://www.biorxiv.org/content/10.1101/2025.11.06.686964v1}
}
```
