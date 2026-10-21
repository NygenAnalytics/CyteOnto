---
description: Compare author cell type labels with the labels of two algorithms using the cyteonto Python package, the precomputed Cell Ontology embeddings and models on Nebius Token Factory.
---

(quickstart-python)=
# Quickstart (Python)

This page compares three author labels with the labels of two algorithms using the
`cyteonto` package. It uses the models that the precomputed ontology files were made
with: Kimi K2.6 to describe labels and Qwen3-Embedding-8B to embed them, both served by
Nebius Token Factory. With these models, CyteOnto loads the ontology embeddings instead
of generating them.

## Before you start

- [Install CyteOnto and download the Cell Ontology files](installation.md).
- Set `NEBIUS_API_KEY` in your environment or in a `.env` file (see
  [API keys](installation.md#installation-api-keys)).
- Optionally, set `CYTEONTO_LOG_LEVEL=INFO` before you start Python. CyteOnto logs at
  the `DEBUG` level by default.

The code below runs in a Jupyter notebook, where you can `await` coroutines directly. To
run it as a script, see [Run it as a script](#quickstart-python-script).

## Create the agent and the model configurations

CyteOnto calls the LLM through a [pydantic-ai](https://ai.pydantic.dev/) agent. Nebius
Token Factory serves an OpenAI-compatible API, so the agent uses pydantic-ai's OpenAI
model with Nebius's base URL.

```python
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cyteonto import CyteOnto, EmbdConfig, LlmConfig

# The LLM that describes labels: Kimi K2.6 on Nebius Token Factory.
agent = Agent(
    OpenAIChatModel(
        "moonshotai/Kimi-K2.6",
        provider=OpenAIProvider(
            api_key=os.environ["NEBIUS_API_KEY"],
            base_url="https://api.tokenfactory.nebius.com/v1/",
        ),
    )
)

# The names that key the caches; they match the precomputed ontology files.
llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")

# The embedding model. Without an apiKey, CyteOnto reads NEBIUS_API_KEY.
embedding = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")
```

`llm.model` must be the model name of the agent; CyteOnto raises a `ValueError` if they
differ. The model names are case-sensitive: with other names, CyteOnto generates the
ontology files for that model pair, which takes thousands of requests (see
[Models and providers](../guides/models-and-providers.md)).

## Load the ontology embeddings

```python
# Load the descriptions and embeddings of the Cell Ontology terms.
cyto = await CyteOnto.from_config(
    agent=agent,
    embedding=embedding,
    llm=llm,
    use_pubmed_tool=False,
)
```

`from_config()` finds the precomputed files of this model pair and calls no model. If you
downloaded the files to another directory with `--data-dir`, pass it as `data_dir`.
`use_pubmed_tool=False` turns off the PubMed search that the agent may otherwise run
while it describes a label, which makes the descriptions faster and cheaper.

## Compare labels

Each algorithm's list has one label per author label, in the same order. The labels at
the same position are compared.

```python
author_labels = [
    "alveolar macrophage",
    "regulatory T cell",
    "CD8-positive, alpha-beta T cell",
]
algorithms = {
    "method_a": ["lung macrophage", "Treg", "CD8 T cell"],
    "method_b": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"],
}

# Describe, embed and match each label, then score each pair.
df = await cyto.compare(
    author_labels=author_labels,
    algorithms=algorithms,
    run_id="quickstart",
)
```

`compare()` asks the LLM whether each distinct label names more than one cell type,
describes the distinct labels of the authors and of each algorithm, embeds the
descriptions and matches them to Cell Ontology terms. For these 9 labels, that is about
18 LLM requests and 9 embedding requests. The `run_id` names the caches of this
comparison, so a second call with the same labels and `run_id` reuses them and calls no
model.

## Read the results

`df` has one row for each algorithm and position, six here:

```python
df[
    [
        "algorithm",
        "author_label",
        "algorithm_label",
        "author_ontology_name",
        "algorithm_ontology_name",
        "cytescore_similarity",
        "similarity_method",
    ]
]
```

The labels are lowercased. `author_ontology_name` and `algorithm_ontology_name` name the
Cell Ontology term that each label matched, and `cytescore_similarity` scores the two
terms: 1 when both labels matched the same term, and lower the less related the terms
are. To compare the algorithms, average their scores:

```python
# Mean score of each algorithm over all label pairs.
df.groupby("algorithm")["cytescore_similarity"].mean()
```

[Read the results](../guides/results.md) explains every column and how to interpret the
scores.

## Check the model usage

```python
# LLM requests and tokens of this instance, and the models that served them.
print(cyto.usage.requests, cyto.usage.totalTokens)
print(cyto.model_pair_usage.to_status_dict())
```

`cyto.usage` counts the requests and tokens of the LLM calls that succeeded; failed
attempts and embedding requests are not counted.

(quickstart-python-script)=
## Run it as a script

Outside a notebook, wrap the calls in a coroutine and run it with `asyncio.run()`. This
script runs the whole quickstart and saves the results:

```python
import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from cyteonto import CyteOnto, EmbdConfig, LlmConfig


async def main() -> None:
    # Load NEBIUS_API_KEY from the .env file next to this script, if there is one.
    load_dotenv()

    # The LLM, and the model names that match the precomputed ontology files.
    agent = Agent(
        OpenAIChatModel(
            "moonshotai/Kimi-K2.6",
            provider=OpenAIProvider(
                api_key=os.environ["NEBIUS_API_KEY"],
                base_url="https://api.tokenfactory.nebius.com/v1/",
            ),
        )
    )
    llm = LlmConfig(provider="nebius", model="moonshotai/Kimi-K2.6")
    embedding = EmbdConfig(provider="nebius", model="Qwen/Qwen3-Embedding-8B")

    # Load the ontology embeddings, compare the labels and save the results.
    cyto = await CyteOnto.from_config(
        agent=agent, embedding=embedding, llm=llm, use_pubmed_tool=False
    )
    df = await cyto.compare(
        author_labels=[
            "alveolar macrophage",
            "regulatory T cell",
            "CD8-positive, alpha-beta T cell",
        ],
        algorithms={
            "method_a": ["lung macrophage", "Treg", "CD8 T cell"],
            "method_b": ["alveolar mac", "T regulatory cell", "cytotoxic T cell"],
        },
        run_id="quickstart",
    )
    df.to_csv("cyteonto_quickstart.csv", index=False)


asyncio.run(main())
```

## Next steps

- [How CyteOnto works](how-cyteonto-works.md): what each step does and what is sent to
  the providers.
- [Models and providers](../guides/models-and-providers.md): other models, fallback
  models and API keys.
- [Similarity metrics](../guides/similarity-metrics.md) and
  [Compound labels](../guides/compound-labels.md): the options of `compare()`.
- [Caching and runs](../guides/caching.md): reuse and delete the caches of a run.
