.PHONY: docker-build docker-up docker-down format check check-all check-mypy pre-commit-install pre-commit-run

docker-build:
	docker compose -f ci_cd/docker/docker-compose.yml build --no-cache backend

docker-up:
	docker compose -f ci_cd/docker/docker-compose.yml up -d

docker-down:
	docker compose -f ci_cd/docker/docker-compose.yml down

format:
	PATH="backend/.venv/bin:$(PATH)" bash ci_cd/scripts/format.sh .

pre-commit-install:
	PATH="backend/.venv/bin:$(PATH)" pre-commit install

pre-commit-run:
	PATH="backend/.venv/bin:$(PATH)" pre-commit run --all-files

check-mypy:
	PATH="backend/.venv/bin:$(PATH)" bash ci_cd/scripts/check_mypy.sh

# Same suite as pre-commit (ruff/mypy/bandit) against backend/.venv.
check-all:
	PATH="backend/.venv/bin:$(PATH)" bash ci_cd/scripts/check_all.sh backend/src

check: check-all
