.PHONY: help install build clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build:  ## Build the package.
	@python setup.py sdist bdist_wheel

install:  ## Install the package.
	@pip install dist/rag_citation-0.0.3-py3-none-any.whl --force-reinstall

clean:  ## Clean up build artifacts.
	@rm -rf dist/ build/ *.egg-info

upload:  
	@twine upload dist/*