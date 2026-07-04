# impurityModel Makefile for Sphinx documentation

# Output directory of the build files
BUILD_CONFIG = _doc_build

# Output directory of the generated documentation
GEN_DOC = generated_doc

# Put it first so that "make" without argument is like "make help".
help:
	sphinx-build -M help a b c

clean:
	rm -rf $(BUILD_CONFIG) $(GEN_DOC)

.PHONY: help clean

# Excluded from the API docs: the test suite (imports pytest, not API) and the
# untracked double_chain_haverkort scratch directory.
$(BUILD_CONFIG)/index.rst:
	sphinx-apidoc -q -f --implicit-namespaces -T -o $(BUILD_CONFIG) ../../src/impurityModel '../../src/impurityModel/test' '../../src/impurityModel/ed/double_chain_haverkort'

# Add makefile as a separate target so that it doesn't match the
# catch-all target. Seems make want to rebuild the Makefile itself as
# part of the process.
Makefile:
	true

# Catch-all target: route all unknown targets to Sphinx using the
# "make mode" option. Warnings are printed but not fatal (no -W): autodoc
# emits unavoidable duplicate-description warnings for documented namedtuple
# attributes (GFUnit/PairwiseGF).
%: $(BUILD_CONFIG)/index.rst
	sphinx-build -M $@ .. $(GEN_DOC) -c . -q
