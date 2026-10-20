---
description: Install the cyteonto Python package from GitHub, download the Cell Ontology files and the precomputed ontology embeddings, and set the API keys of your LLM and embedding providers.
---

(installation)=
# Installation

Install CyteOnto to run comparisons on your own computer. The
[hosted API](quickstart-api.md) needs no installation.

## Requirements

- Python 3.13 or newer.
- About 120 MB of disk space for the Cell Ontology files and the precomputed ontology
  embeddings, and some more for the caches of your runs.
- API keys for an LLM provider and an embedding provider. The precomputed files were
  made with models on Nebius Token Factory, so with a Nebius API key you can start
  without generating any ontology files (see [API keys](#installation-api-keys)).

## Install the package

`cyteonto` is not published on PyPI. Install it from GitHub:

::::{tab-set}
:::{tab-item} pip
:sync: pip
```bash
pip install "git+https://github.com/NygenAnalytics/CyteOnto.git"
```
:::
:::{tab-item} uv
:sync: uv
```bash
# Add CyteOnto to a uv project
uv add "git+https://github.com/NygenAnalytics/CyteOnto.git"

# Or install it into the active virtual environment
uv pip install "git+https://github.com/NygenAnalytics/CyteOnto.git"
```
:::
::::

The package also installs the tools of the hosted service and of its development, such
as FastAPI, Modal, Jupyter and pytest, so install it into a dedicated virtual
environment.

To work on CyteOnto itself, or to run the notebooks in its `notebooks` folder, clone the
repository and install its locked environment instead:

```bash
git clone https://github.com/NygenAnalytics/CyteOnto.git
cd CyteOnto
uv sync
```

(installation-data)=
## Download the Cell Ontology files

CyteOnto needs the Cell Ontology and the descriptions and embeddings of its terms.
Download them once:

```bash
python -m cyteonto.setup
```

The command writes into the `data` directory inside the installed package, or into
another directory that you pass with `--data-dir`. Files that exist are skipped; pass
`--force` to download them again.

| File | Size | Contents |
|---|---|---|
| `cell_ontology/cell_to_cell_ontology.csv` | 0.2 MB | The labels of the Cell Ontology terms: 6,156 labels of 2,902 terms |
| `cell_ontology/cell_to_cell_ontology_enriched.csv` | 0.4 MB | The labels without duplicates, 6,155, with a lowercase `label_normalized` column; used for lookups |
| `cell_ontology/cl.owl` | 63 MB | The Cell Ontology, for the hierarchy metrics |
| `embedding/descriptions/descriptions_nebius_moonshotai-Kimi-K2.6.json` | 2.1 MB | A description of every term, written by Kimi K2.6 |
| `embedding/cell_ontology/embeddings_nebius_moonshotai-Kimi-K2.6_nebius_Qwen-Qwen3-Embedding-8B.npz` | 25 MB | The embedding of every description, made with Qwen3-Embedding-8B |

The command also downloads the descriptions and embeddings of the fallback model pair
(27 MB); if those downloads fail, it warns and continues. If the enriched CSV cannot be
downloaded, it builds it from the original CSV. The command exits with status 1 when a
required file cannot be downloaded.

:::{tip}
Keep the files outside the installed package, so that reinstalling or removing the
package never touches them:

```bash
python -m cyteonto.setup --data-dir ~/cyteonto-data
```

Then pass the same directory to CyteOnto as `data_dir`. CyteOnto does not expand `~`
in paths, so pass `Path.home() / "cyteonto-data"`.
:::

[Models and providers](../guides/models-and-providers.md) explains which models the
precomputed files belong to, and [Data files and formats](../reference/files.md)
describes every file.

(installation-api-keys)=
## API keys

CyteOnto calls two models, and each needs a key from its provider.

- **The LLM** is called through a pydantic-ai agent that you create, and you give the
  agent its key. The quickstart reads it from `NEBIUS_API_KEY`.
- **The embedding model** is called by CyteOnto. It uses `EmbdConfig(apiKey=...)` when
  you set it, and otherwise reads the environment variable of the provider:

| Embedding provider | Environment variable |
|---|---|
| `deepinfra` | `DEEPINFRA_API_KEY` |
| `fireworks` | `FIREWORKS_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `nebius` | `NEBIUS_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `together` | `TOGETHER_API_KEY` |

If that variable is not set, CyteOnto uses `EMBEDDING_MODEL_API_KEY`. The `ollama`
provider, a local Ollama server, needs no key. Export the variables in your shell, or keep
them in a `.env` file:

```bash
# .env
NEBIUS_API_KEY=<your-nebius-api-key>
```

- **In a notebook**, `import cyteonto` loads the `.env` file of the working directory, or
  of the nearest parent directory that has one.
- **In a script**, call `load_dotenv()` from the `dotenv` package, which is installed with
  `cyteonto`, to load the `.env` file next to the script. On its own, `import cyteonto`
  only searches the directories above the installed package.

Loading a `.env` file does not change variables that are already set. Variables that
CyteOnto reads when it is imported, such as `CYTEONTO_LOG_LEVEL`, must be set before
`import cyteonto`. Keep `.env` files out of version control.
[Environment and limits](../reference/configuration.md) lists all the variables that
CyteOnto reads.

## Check the installation

```bash
python -c "import cyteonto; print(cyteonto.__version__)"
```

This prints the installed version, such as `0.3.0`.

## Update

Run the install command again with `--force-reinstall`, or, in a clone of the
repository, run `git pull` and then `uv sync`. Downloaded files in the package's `data`
directory may be lost when the package is reinstalled; run `python -m cyteonto.setup`
again if they are missing.

## Next step

Compare your first labels in Python: [Quickstart (Python)](quickstart-python.md).
