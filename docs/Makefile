UVRUN ?= uv run --group docs

.PHONY: help html check-reference

help:
	@echo 'make html             Build the site and the API reference'
	@echo 'make check-reference  Build and check the Sphinx API'

check-reference:
	@rm -rf build/api build/api-text
	@$(UVRUN) sphinx-build -b html -n -W source/reference/api build/api
	@$(UVRUN) python check_reference_coverage.py
	@$(UVRUN) sphinx-build -b text -n -W -d build/api/.doctrees source/reference/api build/api-text

html: check-reference
	@cd .. && $(UVRUN) python -m docs.build_docs
