# CI
.PHONY: checks
checks:
	poetry run pre-commit run --all-files
