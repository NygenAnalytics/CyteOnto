"""Structure checks for the documentation sources in docs/source."""

import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest

yaml = pytest.importorskip("yaml")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DOCS_SOURCE = _REPOSITORY_ROOT / "docs" / "source"
_API_SOURCE = _DOCS_SOURCE / "reference" / "api"
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_AUTODOC_TARGET = re.compile(
    r"(?m)^(?:\.\. (?:auto\w+|py:\w+)::|[`:]{3,}\{(?:auto\w+|py:\w+)\})[ \t]*([^\s(]+)"
)
_TAB_SET = re.compile(
    r"(?ms)^(?P<fence>:{4,}|`{4,})\{tab-set\}[^\n]*\n(?P<body>.*?)^(?P=fence)[ \t]*$"
)
_TAB_ITEM = re.compile(
    r"(?m)^(?::{3,}|`{3,})\{tab-item\}[ \t]*(?P<title>[^\n]*)\n"
    r"(?P<options>(?:[ \t]*:[\w-]+:[^\n]*\n)*)"
)
# Tabs with the same sync key switch together on every page.
_TAB_SYNC = {"curl": "curl", "Python": "python", "pip": "pip", "uv": "uv"}
_SECTIONS = ["Getting started", "Guides", "Reference", "Support"]
_FIELD = re.compile(r"(?m)^   \.\. py:attribute:: (\w+)\n((?:      :\w+:.*\n)*)")
_TABLE_ROW = re.compile(r"(?m)^\| `(\w+)` \| [^|]+ \| ([^|]+) \|")


def _project() -> dict[str, Any]:
    project: dict[str, Any] = yaml.safe_load((_DOCS_SOURCE / "myst.yml").read_text())[
        "project"
    ]
    return project


def _notebook_cells(path: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = json.loads(path.read_text())["cells"]
    for cell in cells:
        if not isinstance(cell["source"], str):
            cell["source"] = "".join(cell["source"])
    return cells


def _source_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    return "\n".join(cell["source"] for cell in _notebook_cells(path))


def _pages() -> list[Path]:
    return sorted(
        path
        for path in _DOCS_SOURCE.rglob("*")
        if path.suffix in {".md", ".ipynb"}
        and not {"_build", ".ipynb_checkpoints"} & set(path.parts)
    )


def _api_pages() -> list[Path]:
    return sorted(_API_SOURCE.glob("*.md"))


def _guide_pages() -> list[Path]:
    """Return the MyST pages; Sphinx builds the pages in reference/api."""
    return [page for page in _pages() if not page.is_relative_to(_API_SOURCE)]


def _toc_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = []
    for entry in entries:
        found.append(entry)
        found.extend(_toc_entries(entry.get("children", [])))
    return found


def _toc_files() -> set[str]:
    return {
        entry["file"] for entry in _toc_entries(_project()["toc"]) if "file" in entry
    }


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _route(page: Path) -> str:
    """Return the MyST route of a page, such as ``getting-started/installation``."""
    relative = page.relative_to(_DOCS_SOURCE).with_suffix("")
    if relative.as_posix() == "index":
        return ""
    return "/".join(_slug(part) for part in relative.parts)


def _route_source(route: str) -> Path:
    """Return the source page published at ``route``, such as ``api/cyteonto.html``."""
    route = route.split("#", maxsplit=1)[0]
    if route.startswith("api/"):
        name = route.removeprefix("api/")
        assert name.endswith(".html"), route
        return _API_SOURCE / name.replace(".html", ".md")
    routes = {_route(page): page for page in _guide_pages()}
    route = "" if route in {"", "./"} else route.strip("/")
    assert route in routes, f"No documentation page publishes {route!r}"
    return routes[route]


def _link_source(page: Path, target: str) -> Path:
    """Return the documentation source that a relative link from ``page`` opens."""
    path = target.split("#", maxsplit=1)[0]
    if path.startswith("/"):
        return _route_source(path.lstrip("/"))
    if page.parent == _API_SOURCE:
        # Sphinx pages are served at /api/<page>.html, beside the MyST routes.
        if path.startswith("../"):
            return _route_source(path.removeprefix("../"))
        return _API_SOURCE / path.replace(".html", ".md")
    resolved = (page.parent / path).resolve()
    if resolved.parent == _API_SOURCE.resolve():
        # The build points links to the Sphinx sources at their /api/ pages.
        assert resolved.suffix in {".html", ".md"}, (page, target)
        return resolved.with_suffix(".md")
    assert path.endswith((".md", ".ipynb")), (page, target)
    return resolved


def test_llms_index_links_to_every_published_page() -> None:
    contents = (_DOCS_SOURCE / "llms.txt").read_text()
    targets = _MARKDOWN_LINK.findall(contents)

    assert targets
    assert contents.startswith("# CyteOnto\n\n> ")
    for target in targets:
        assert "://" not in target, target
        # The build maps these exact routes to the plain-text exports.
        assert target == "./" or target.endswith(("/", ".html")), target
        source = _route_source(target)
        assert source.resolve().is_relative_to(_DOCS_SOURCE.resolve())
        assert source.is_file(), target
    listed = {_route_source(target).resolve() for target in targets}
    published = {(_DOCS_SOURCE / page).resolve() for page in _toc_files()}
    published |= {page.resolve() for page in _api_pages()}
    assert published - listed == set(), "Pages missing from llms.txt"


def test_navigation_publishes_every_page() -> None:
    project = _project()
    toctree = project["toc"]
    sections = {
        entry["title"]: entry["children"] for entry in toctree if "children" in entry
    }

    assert toctree[0] == {"file": "index.md"}
    assert list(sections) == _SECTIONS
    assert sections["Reference"][0] == {
        "title": "Python API",
        "url": "/api/index.html",
        "open_in_same_tab": True,
    }
    issue_trackers = {entry["url"] for entry in sections["Support"] if "url" in entry}
    assert issue_trackers == {"https://github.com/NygenAnalytics/CyteOnto/issues"}
    assert "llms.txt" in project["static_files"]
    assert "../build/api" not in project["static_files"]
    assert "reference/api/**" in project["exclude"]

    published = _toc_files()
    for file in published:
        assert (_DOCS_SOURCE / file).is_file(), f"{file} is in the TOC but missing"
    for page in _guide_pages():
        relative = page.relative_to(_DOCS_SOURCE).as_posix()
        assert relative in published, f"{relative} is not in the TOC"


def test_relative_links_resolve_to_documentation_pages() -> None:
    for page in _pages():
        for target in _MARKDOWN_LINK.findall(_source_text(page)):
            if "://" in target or target.startswith(("#", "mailto:")):
                continue
            source = _link_source(page, target)
            assert source.is_file(), (page, target)
            assert source.resolve().is_relative_to(_DOCS_SOURCE.resolve())


def test_pages_have_a_description_and_one_title() -> None:
    for page in _guide_pages():
        if page.suffix == ".ipynb":
            metadata = json.loads(page.read_text())["metadata"]
            description = metadata.get("description")
            text = "\n".join(
                cell["source"]
                for cell in _notebook_cells(page)
                if cell["cell_type"] == "markdown"
            )
        else:
            text = page.read_text()
            frontmatter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
            assert frontmatter, f"{page} has no front matter"
            description = (yaml.safe_load(frontmatter[1]) or {}).get("description")
            text = text[frontmatter.end() :]
        assert isinstance(description, str) and description.strip(), page
        text = re.sub(r"(?ms)^(`{3,}|~{3,}).*?^\1", "", text)
        assert len(re.findall(r"(?m)^# ", text)) == 1, f"{page} needs one H1 title"


def test_autodoc_targets_use_the_public_cyteonto_names() -> None:
    import cyteonto

    # Document objects under the names users import them by, not their modules.
    targets = []
    for page in _api_pages():
        for target in _AUTODOC_TARGET.findall(page.read_text()):
            module, _, name = target.rpartition(".")
            assert module == "cyteonto" and name in cyteonto.__all__, (page, target)
            targets.append(target)
    assert sorted(targets) == sorted(f"cyteonto.{name}" for name in cyteonto.__all__)


def test_tabs_are_synced_across_pages() -> None:
    for page in _pages():
        text = _source_text(page)
        tab_sets = list(_TAB_SET.finditer(text))
        assert len(tab_sets) == text.count("{tab-set}"), (page, "unparsed tab-set")
        items = 0
        for tab_set in tab_sets:
            for item in _TAB_ITEM.finditer(tab_set["body"]):
                title = item["title"].strip()
                options = dict(
                    re.findall(
                        r"(?m)^[ \t]*:([\w-]+):[ \t]*(.*?)[ \t]*$", item["options"]
                    )
                )
                assert title in _TAB_SYNC, (page, title)
                assert options.get("sync") == _TAB_SYNC[title], (page, title)
                items += 1
        assert items == text.count("{tab-item}"), (page, "unparsed tab-item")


def test_pages_contain_no_html_comments() -> None:
    # The site publishes page sources, page data and the search index, so comments leak.
    for page in _pages():
        assert "<!--" not in _source_text(page), page


def _exported_models() -> list[str]:
    import cyteonto
    from pydantic import BaseModel

    return [
        name
        for name in cyteonto.__all__
        if isinstance(getattr(cyteonto, name), type)
        and issubclass(getattr(cyteonto, name), BaseModel)
    ]


def _documented_names(text: str) -> set[str]:
    """Return the objects that an API page documents, such as ``cyteonto.X.y``."""
    names: set[str] = set()
    current = None
    for line in text.splitlines():
        if match := re.match(r"^\.\. auto\w+:: (cyteonto\.\w+)", line):
            current = match[1]
            names.add(current)
        elif current and (match := re.match(r"^   :members: (.+)$", line)):
            names.update(f"{current}.{name.strip()}" for name in match[1].split(","))
        elif current and (match := re.match(r"^   \.\. py:\w+:: (\w+)", line)):
            names.add(f"{current}.{match[1]}")
    return names


def test_api_anchors_name_documented_objects() -> None:
    documented = {
        page.resolve(): _documented_names(page.read_text()) for page in _api_pages()
    }
    checked = 0
    for page in _pages():
        for target in _MARKDOWN_LINK.findall(_source_text(page)):
            fragment = target.partition("#")[2]
            if "://" in target or not fragment.startswith("cyteonto."):
                continue
            source = _link_source(page, target).resolve()
            assert fragment in documented.get(source, set()), (page, target)
            checked += 1
    assert checked


@pytest.mark.parametrize("name", _exported_models())
def test_model_fields_are_documented_with_their_defaults(name: str) -> None:
    import cyteonto

    model = getattr(cyteonto, name)
    pages = [
        page.read_text()
        for page in _api_pages()
        if f".. autoclass:: cyteonto.{name}\n" in page.read_text()
    ]
    assert len(pages) == 1, f"cyteonto.{name} needs one API page"
    text = pages[0]
    start = text.index(f".. autoclass:: cyteonto.{name}\n")
    end = text.find("\n.. autoclass::", start + 1)
    documented = _FIELD.findall(text[start : end if end != -1 else None])

    assert [field for field, _ in documented] == list(model.model_fields)
    for field, options in documented:
        values = dict(re.findall(r":(\w+): (.*)", options))
        info = model.model_fields[field]
        if info.is_required():
            assert "value" not in values, (name, field)
        else:
            default = info.get_default(call_default_factory=True)
            assert values.get("value") == repr(default), (name, field)


def test_http_api_reference_matches_the_request_schema() -> None:
    from modal_app.models import CompareRequest, EmbdProvider, LlmProvider

    text = (_DOCS_SOURCE / "reference" / "http-api.md").read_text()
    section = text[text.index("## `POST /compare`") : text.index("## `GET /status")]
    rows = _TABLE_ROW.findall(section)

    assert [field for field, _ in rows] == list(CompareRequest.model_fields)
    for field, default in rows:
        info = CompareRequest.model_fields[field]
        if info.is_required():
            assert default.strip() == "Required", field
        else:
            value = json.dumps(info.get_default(call_default_factory=True))
            assert default.strip() == f"`{value.strip(chr(34))}`", field
    for provider in (*get_args(LlmProvider), *get_args(EmbdProvider)):
        assert f"`{provider}`" in text, provider


def test_embedding_provider_table_matches_the_endpoints() -> None:
    from cyteonto.config import Config

    text = (_DOCS_SOURCE / "guides" / "models-and-providers.md").read_text()
    rows = re.findall(r"(?m)^\| `(\w+)` \| `(https?://[^`]+)` \| `(\w+)` \|$", text)
    working = {
        provider: (url, Config.PROVIDER_API_KEY_ENV[provider])
        for provider, url in Config._PROVIDER_URL.items()
        if provider != "ollama"
    }

    assert {provider: (url, env) for provider, url, env in rows} == working


def test_result_columns_are_documented_in_order() -> None:
    from cyteonto.config import Config

    text = (_DOCS_SOURCE / "guides" / "results.md").read_text()
    section = text[text.index("## Columns") : text.index("## `similarity_method`")]

    assert re.findall(r"(?m)^\| `(\w+)` \|", section) == Config.RESULT_COLUMNS
