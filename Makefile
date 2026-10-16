SHELL:=/usr/bin/env bash

.PHONY: format
format:
	black .

.PHONY: lint
lint:
	-mypy test dmiw
	-flake8 dmiw --count --statistics

.PHONY: unit
unit:
	pytest --cov=dmiw

.PHONY: package
package:
	pip check

.PHONY: test
test: format lint unit package
