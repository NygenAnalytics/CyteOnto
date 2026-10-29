# Documentation instructions

These instructions cover the documentation sources and site builds. The site documents the `cyteonto` Python package and the hosted HTTP API, and is published at `https://docs.nygen.io/CyteOnto/`.

- Write pages in Markdown in `docs/source/`. Check the code in the pages against `cyteonto/` for the package, and `modal_app/models.py`, `modal_app/api.py` and `modal_app/worker.py` for the HTTP API.
- MyST builds the pages. Sphinx builds only `docs/source/reference/api/`, which the site serves at `/api/`.
- Import the package as `import cyteonto` or `from cyteonto import ...`, and use `cyteonto.*` autodoc targets, the names under which users import each object.
- Document the fields of the pydantic models as `py:attribute` entries inside their `autoclass` directive, in field order, with a `:value:` for each optional field; autodoc does not list pydantic fields.
- Start every page with front matter holding one `description:` line, then a label and one H1 title. Add every page to the table of contents in `docs/source/myst.yml` and to `docs/source/llms.txt`, with a one-line summary.
- The reference scores in `guides/similarity-metrics.md` and `guides/results.md` and the description in `getting-started/how-cyteonto-works.md` come from the precomputed files of the default model pair. Recompute them when those files change.

Avoid math markup, which the plain-text exports render three times; write formulas as code.

## Tabs

Show alternatives that a reader chooses once in synced tabs, so that the choice carries across pages: `curl` and `Python` for the HTTP API, and `pip` and `uv` for installation.

````markdown
::::{tab-set}
:::{tab-item} curl
:sync: curl
```bash
curl -sS https://cyteonto.nygen.io/health
```
:::
:::{tab-item} Python
:sync: python
```python
requests.get("https://cyteonto.nygen.io/health", timeout=30).json()
```
:::
::::
````

## Links

- Link other pages with relative file links, such as `[Installation](../getting-started/installation.md)`. Relative links must resolve to a page in `docs/source/`.
- Link the Python API through its Sphinx source with an `.html` (or `.md`) suffix, such as `[compare()](../reference/api/cyteonto.html#cyteonto.CyteOnto.compare)`. The build points these links at `/api/cyteonto.html#cyteonto.CyteOnto.compare`. Anchors are the documented names, such as `#cyteonto.CyteOnto.compare` or `#cyteonto.EmbdConfig.provider`. The absolute form `/api/<page>.html` also works.
- In the Sphinx pages, link guides relative to `/api/`, such as `[Python quickstart](../getting-started/quickstart-python)`.

## Setup and rendering

Use Node.js 24 and install the Python tools:

```bash
uv sync --group docs
make -C docs html
```

`make -C docs html` builds the API reference with Sphinx, nitpicky and with warnings as errors, and checks its coverage. It then stages `docs/source` in `docs/build/site`, builds the site with `myst build --html --strict`, and writes a plain-text export of every page, `llms.txt` with links to the exports, the sitemap and canonical links. The site is written to ignored `docs/build/site/_build/html/`. To view a build, serve the output directory, for example with `python -m http.server -d docs/build/site/_build/html 8000`, and open `http://localhost:8000/`.

The build also accepts executed Jupyter notebooks with saved outputs and a code fingerprint, as the CyteType documentation does, but CyteOnto has no runner that executes and saves them. The notebooks in `notebooks/` are not part of the site.

Pull requests and the Pages workflow run the documentation tests and `make -C docs html`. GitHub Pages publishes from `master` at `https://docs.nygen.io/CyteOnto/`. The workflow builds with `BASE_URL=/CyteOnto` and uploads the generated site as the Pages artifact; the organization's Pages site, `NygenAnalytics/NygenAnalytics.github.io`, serves it under `/CyteOnto/` and owns the root `robots.txt`. Keep rendered site files outside Git.

The build also writes a `robots.txt` into the site, as CyteType's build does. Crawlers read only the root `robots.txt` of `docs.nygen.io`, which lists the sitemap of each project, so the copy under `/CyteOnto/` has no effect.

## Checks

```bash
uv run --group docs pytest tests/test_docs_structure.py tests/test_docs_render.py
make -C docs check-reference
```

The Docs workflow also lints `docs/` and the documentation tests with ruff, because the Tests workflow skips documentation-only changes. The API check is a nitpicky, warnings-as-errors Sphinx build. Its inventory must document every `cyteonto.__all__` export under its public `cyteonto.` name, and every public method and property of the exported classes exactly once. The structure tests check the table of contents, `llms.txt`, relative links, tabs, and that pages hold no HTML comments. They also check the documented model fields and defaults against the pydantic models, the HTTP API reference against `modal_app.models.CompareRequest`, the embedding provider table against `cyteonto.config.Config`, and the result columns against `Config.RESULT_COLUMNS`. Inspect the rendered pages after changing their structure.
