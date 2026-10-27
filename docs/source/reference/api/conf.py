import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT.parents[1]))

project = "CyteOnto API"
copyright = "2025-2026, Nygen Analytics AB"
author = "Nygen Analytics AB"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]
master_doc = "index"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
myst_enable_extensions = ["colon_fence"]
myst_all_links_external = True
pygments_style = "sphinx"
language = "en"
# Render single-backtick text in docstrings as code, not as a cross-reference.
default_role = "literal"

# Show each class's __init__ arguments with its description.
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
# Render "str, optional" in Google-style argument types as text, not as references.
napoleon_preprocess_types = True
napoleon_use_rtype = False
typehints_defaults = "comma"
# Write unions as "str | None", as the code does, instead of Optional[str].
always_use_bars_union = True

html_theme = "sphinx_book_theme"
templates_path = ["_templates"]
html_sidebars = {"**": ["docs-navigation.html"]}
html_css_files = ["https://docs.nygen.io/assets/brand/docs.css"]
html_logo = "https://docs.nygen.io/assets/brand/logo/cyteonto.svg"
html_favicon = "https://docs.nygen.io/assets/brand/logo/cyteonto.ico"
html_title = "CyteOnto API reference"
html_baseurl = "https://docs.nygen.io/CyteOnto/api/"
html_theme_options = {
    "logo": {"text": "CyteOnto", "alt_text": "CyteOnto"},
    "repository_url": "https://github.com/NygenAnalytics/CyteOnto",
    "path_to_docs": "docs/source/reference/api",
    "article_header_start": [],
    "article_header_end": ["toggle-secondary-sidebar.html"],
    "show_prev_next": False,
    "footer_content_items": ["extra-footer.html"],
    "extra_footer": '<a class="docs-home-link" href="https://docs.nygen.io/"><img class="brand-symbol" src="https://docs.nygen.io/assets/brand/logo/nygen-symbol.svg" alt="Nygen documentation home" width="32" height="32"></a>',
    "show_navbar_depth": 2,
    "navigation_with_keys": False,
}

# The reference names these types without linking to their documentation.
nitpick_ignore = [
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "pathlib.Path"),
    ("py:class", "pydantic_ai.agent.Agent"),
    ("py:data", "typing.Any"),
    ("py:data", "typing.Literal"),
]
