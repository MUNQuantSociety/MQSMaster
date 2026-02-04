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
5. Check DB secrets and run tests only if DB credentials are available.

### Database-gated tests

Tests that require the database only run when all required secrets are present:
- DB_HOST
- DB_PORT
- DB_NAME
- DB_USER
- DB_PASSWORD
- DB_SSLMODE (optional but supported)

If secrets are missing, the workflow skips pytest to avoid failing CI.

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
   - pip install ruff pytest
3. Export the same DB_* environment variables used in CI if your tests need database access.
4. Run tests:
   - python -m pytest -q

## Lint locally

- python -m ruff check src tests scripts --select E9,F63,F7,F82

## Run Docker locally (optional)

1. Build the image from the repo root:
   - docker build -t mqs-bot:local .
2. Run the container (add env vars if needed):
   - docker run --rm -p 8000:8000 mqs-bot:local

If your app expects DB credentials or other secrets, pass them with `-e` flags or an env file.
