# CyteOnto
 
**Semantic Cell Type Annotation Comparison Using Large Language Models and Cell Ontology**
 
CyteOnto helps researchers benchmark cell type annotation algorithms by measuring semantic similarity between annotation labels, leveraging the structured knowledge of Cell Ontology (CL) and large language models
 
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
 
---
 
## Installation
 
### Prerequisites
- Python 3.12+
- UV package manager (recommended)
 
### Install
```bash
# Clone the repository
git clone https://github.com/NygenAnalytics/CyteOnto.git
cd CyteOnto
 
# Setup environment
uv sync
```
 
## Quick start

You can follow the tutorial in [quick_tutorial.ipynb](./notebooks/quick_tutorial.ipynb)
 
### 1. Set API Keys as Environment Variables
 
```bash
LLM_API_KEY=your_api_key_here               # For example OpenAI (can be other like groq, openrouter, google, xai, deepinfra, etc.)
EMBEDDING_MODEL_API_KEY=your_api_key_here   # Can be the same as above if the embedding model is from the same provider
 
# Optional: for higher rate limits
NCBI_API_KEY=your_ncbi_api_key_here         # for using pubmed tool calls
```
 
### 2. Download precomputed embeddings

```bash
uv run python scripts/show_embeddings.py
## ⬇️  moonshot-ai_kimi-k2 (Recommended)
##     Name: Moonshot AI Kimi-K2 (descriptions) + Qwen3-Embedding-8B (embeddings)
##     Status: Not downloaded
## ⬇️  deepseek_v3
##     Name: DeepSeek V3 (descriptions) + Qwen3-Embedding-8B (embeddings)
##     Status: Not downloaded
 
uv run python scripts/download_embedding.py moonshot-ai_kimi-k2
```
 
### 3. Setup LLM Agent
 
```python
import cyteonto
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
 
# Initialize your LLM agent
model = OpenAIModel(
    "moonshotai/Kimi-K2-Instruct",
    provider=OpenAIProvider(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=os.getenv("LLM_API_KEY"),
    ),
)
agent = Agent(model)
```
 
### 4. Initialize CyteOnto with LLM and Embedding model
```python
cyto = cyteonto.CyteOnto(
    base_agent=agent,
    embedding_model="Qwen/Qwen3-Embedding-8B", 
    embedding_provider="deepinfra"
)
```
 
### 5. Compare Cell Type Annotations
```python
# Your data
author_labels = ["animal stem cell", "BFU-E", "CFU-M", "neutrophilic granuloblast"]
algorithm1_labels = ["stem cell", "blast forming unit erythroid", "erythroid stem cell", "spermatogonium"]
algorithm2_labels = ["neuronal receptor cell", "stem cell", "smooth muscle cell", "ovum"]
 
# Perform batch comparison
results_df = await cyto.compare_batch(
    study_name="sample_study",          # Save and cache all the results to this directory. Serves as a unique run id.
    author_labels=author_labels,
    algo_comparison_data=[
        ("algorithm1", algorithm1_labels),
        ("algorithm2", algorithm2_labels)
    ],
)
print(results_df)
```
| study_name   | algorithm  | pair_index | author_label               | algorithm_label                          | author_ontology_id | author_embedding_similarity | algorithm_ontology_id  | algorithm_embedding_similarity | ontology_hierarchy_similarity| similarity_method      |
|--------------|------------|------------|----------------------------|------------------------------------------|--------------------|-----------------------------|------------------------|-------------------------------|-------------------------------|------------------------|
| sample_study | algorithm1 | 0          | animal stem cell           | stem cell                                | CL:0000034         | 0.9169                      | CL:0000723             | 0.8233                        | 0.9868                        | ontology_hierarchy     |
| sample_study | algorithm1 | 1          | BFU-E                      | blast forming unit erythroid             | CL:0001066         | 0.8798                      | CL:0001066             | 0.8651                        | 1.0000                        | ontology_hierarchy     |
| sample_study | algorithm1 | 2          | CFU-M                      | erythroid stem cell                      | CL:0000049         | 0.7403                      | CL:0000038             | 0.8797                        | 0.9121                        | ontology_hierarchy     |
| sample_study | algorithm1 | 3          | neutrophilic granuloblast  | spermatogonium                           | CL:0000042         | 0.9121                      | CL:0000020             | 0.9056                        | 0.7566                        | ontology_hierarchy     |
| sample_study | algorithm2 | 0          | animal stem cell           | neuronal receptor cell                   | CL:0000034         | 0.9169                      | CL:0000006             | 0.9087                        | 0.8564                        | ontology_hierarchy     |
| sample_study | algorithm2 | 1          | BFU-E                      | stem cell                                | CL:0001066         | 0.8798                      | CL:0000037             | 0.8089                        | 0.9234                        | ontology_hierarchy     |
| sample_study | algorithm2 | 2          | CFU-M                      | smooth muscle cell                       | CL:0000049         | 0.7403                      | CL:0000027             | 0.9222                        | 0.8703                        | ontology_hierarchy     |
| sample_study | algorithm2 | 3          | neutrophilic granuloblast  | ovum                                     | CL:0000042         | 0.9121                      | CL:0000025             | 0.9099                        | 0.8222                        | ontology_hierarchy     |

---
 
## Understanding Result Dataframe
 
### Column Definitions
- **`study_name`**: The study name used to save and cache the run's results and metadata; serves as a unique run identifier (string).
- **`algorithm`**: The name or identifier of the annotation algorithm being evaluated (e.g., "algorithm1").
- **`pair_index`**: Zero-based index for the author/algorithm label pair within the batch run.
- **`author_label`**: Original label provided by the study author for a given cell type (string).
- **`algorithm_label`**: Label assigned by the evaluated algorithm for the same item (string).
- **`author_ontology_id`**: Cell Ontology (CL) identifier matched to the author label, or null if no match (e.g., "CL:0000548").
- **`author_embedding_similarity`**: Embedding-based similarity score (0–1) between the author label and its matched ontology term; null if no ontology match.
- **`algorithm_ontology_id`**: Cell Ontology (CL) identifier matched to the algorithm label, or null if no match.
- **`algorithm_embedding_similarity`**: Embedding-based similarity score (0–1) between the algorithm label and its matched ontology term; null if no ontology match.
- **`ontology_hierarchy_similarity`**: Final similarity (0–1) computed between the two matched ontology terms based on CL structure; may be null if one or both labels lack ontology matches.
- **`similarity_method`**: Method used to determine the row’s final similarity value. Possible values:
    - `ontology_hierarchy`: both labels matched CL terms and similarity uses ontology structure,
    - `string_similarity`: fallback using string similarity (SequenceMatcher) when CL matches are missing,
    - `partial_match`: only one label found a CL match,
    - `no_matches`: neither label matched CL and no similarity could be computed.
---

## Advanced Usage and Information

Refer to [adv_tutorial.ipynb](./notebooks/adv_tutorial.ipynb) for advanced usage, custom embedding generation, custom file handling, and more.

For more detailed information, consult the project's documentation files:
- [Workflow](./docs/WORKFLOW.md): Provides an in-depth look at the internal processing pipeline
- [File management](./docs/FILE_MANAGEMENT.md): Explains how to manage input and output files

---
 
## Contributing
 
We welcome contributions! Please see our contribution guidelines for details.
 
### Development Setup
```bash
git clone https://github.com/NygenAnalytics/CyteOnto.git
cd CyteOnto
uv sync --dev
```
 

### Running Tests

Run tests using the following commands:

```bash
uv run pytest tests/
uv run pytest tests/ -v --cov=cyteonto # coverage report
```

More details can be found in the [tests README](./tests/README.md).

---
