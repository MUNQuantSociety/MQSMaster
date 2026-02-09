# CI/CD Overview

This repository uses GitHub Actions to run CI on pull requests to main and CD on pushes to main.

## Triggers

- Pull requests to main: runs CI checks.
- Pushes to main: runs CI checks, then builds and publishes the Docker image if a Dockerfile is present.

## End-to-end flow

PR to main → CI (lint + tests)
Push to main → CI → CD (if CI passes)
Docker image built and pushed (only on push to main with Dockerfile)

## CI workflow (job: ci)

Steps:
1. Checkout repository.
2. Set up Python 3.11 with pip caching.
3. Install dependencies (project + test/lint tools).
4. Lint with Ruff (fast error checks).
5. Check DB secrets and run unit tests with coverage.
6. Run DB-gated integration tests when credentials are available.
7. Enforce per-file coverage minimum (50%).
8. Fail on warnings and deprecations in CI.

### Database-gated tests

Tests that require the database only run when all required secrets are present:
- DB_HOST
- DB_PORT
- DB_NAME
- DB_USER
- DB_PASSWORD
- DB_SSLMODE (optional but supported)

If secrets are missing, the workflow only runs unit tests; DB tests are skipped.

### Coverage and per-file enforcement

CI generates coverage reports and fails if any file drops below 50% coverage.
The check runs against the coverage JSON produced by pytest-cov.

### Warnings and deprecations

CI treats warnings as errors to keep signal clean. Locally, deprecation warnings
are also treated as errors via pytest configuration.

## CD workflow (job: build-and-publish)

- Runs only on pushes to main.
- Builds and publishes a Docker image to GitHub Container Registry (GHCR).
- Skips publishing if Dockerfile is not found.

Steps:
1. Check for Dockerfile at repo root.
2. Log in to GHCR.
3. Build and push the image.
4. Tags include `latest` and the commit SHA via metadata action.

## How to run tests locally

1. Create and activate a virtual environment.
2. Install dependencies:
   - pip install -r requirements.txt
   - pip install -e .
   - pip install ruff pytest pytest-cov
3. Export the same DB_* environment variables used in CI if your tests need database access.
4. Run tests:
   - python -m pytest -q -m "not db" --cov=. --cov-report=term-missing --cov-report=json
   - python scripts/check_per_file_coverage.py --min 50 --coverage-file coverage.json

If you have DB credentials and want to run DB-gated tests too:
   - python -m pytest -q -m "db" --cov=. --cov-append --cov-report=term-missing --cov-report=json
   - python scripts/check_per_file_coverage.py --min 50 --coverage-file coverage.json

### Cache behavior

CI caches pip and pytest to speed up repeated runs. Locally, pytest writes cache
to `.pytest_cache` automatically.

## Lint locally

- python -m ruff check src tests scripts --select E9,F63,F7,F82

## Run Docker locally (optional)

1. Build the image from the repo root:
   - docker build -t mqs-bot:local .
2. Run the container (add env vars if needed):
   - docker run --rm -p 8000:8000 mqs-bot:local

If your app expects DB credentials or other secrets, pass them with `-e` flags or an env file.
