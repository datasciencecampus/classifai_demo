.PHONY: all
all: ## Show the available make targets.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep

setup-git-hooks: ## build & add pre-commit and pre-push hooks
	pre-commit install --hook-type pre-commit --hook-type pre-push