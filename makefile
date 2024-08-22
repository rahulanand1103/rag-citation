.PHONY: help install build clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

build:  ## Build the package.
	@python setup.py sdist bdist_wheel

clean:  ## Clean up build artifacts.
	@rm -rf dist/ build/ *.egg-info