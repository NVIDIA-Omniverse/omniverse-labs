# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Project root (absolute so paths work after cd)
PROJECT_ROOT  = $(abspath ..)

# Python package directory (where pyproject.toml lives)
PYTHON_DIR    = $(PROJECT_ROOT)/python

# uv command and docs dependencies (cd into python/ for pyproject.toml, use absolute paths for sphinx)
UV            ?= uv
UVRUN         = cd $(PYTHON_DIR) && $(UV) run --extra docs

# Sphinx source and build dirs as absolute paths (needed because UVRUN changes cwd)
DOCS_DIR      = $(PROJECT_ROOT)/docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(UVRUN) $(SPHINXBUILD) -M help "$(DOCS_DIR)" "$(DOCS_DIR)/$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile doxygen clean html

# Run Doxygen to generate XML for Breathe
doxygen:
	@echo "Running Doxygen..."
	@mkdir -p _doxygen
	@doxygen Doxyfile

# Clean build artifacts
clean:
	@rm -rf $(BUILDDIR)
	@rm -rf _doxygen

# Build HTML with Doxygen first, using uv for dependencies
html: doxygen
	$(UVRUN) $(SPHINXBUILD) -M html "$(DOCS_DIR)" "$(DOCS_DIR)/$(BUILDDIR)" $(SPHINXOPTS) $(O)
