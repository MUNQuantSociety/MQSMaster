# Base image aligned with CI Python version
FROM python:3.12-slim AS build

#install uv from official source
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory inside the container
WORKDIR /app
# Keep Python logs unbuffered and avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Copy dependency files first for faster builds
COPY pyproject.toml requirements.txt ./

# Install system build deps, uv and Python dependencies
RUN uv venv MQS && \
    . MQS/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install uv && \
    uv pip install --no-cache-dir --only-binary :all: -r requirements.txt
# Copy the rest of the project
COPY . /app
RUN . MQS/bin/activate && \
    uv pip install -e .


FROM python:3.12-slim AS runtime

ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# Add the venv’s bin directory to PATH so its python and scripts are used
ENV PATH="/app/MQS/bin:$PATH"

# start.sh's preflight requires curl (market-hours check against FMP) and jq
# (parsing that response) -- python:3.12-slim ships neither.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl jq && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -d /app -s /bin/bash appuser

# start.sh shells out to curl (FMP market-hours check) and jq (parsing the response)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build --chown=appuser:appgroup /app .

USER appuser
# Default command (runs the live trading entrypoint)
CMD ["bash", "start.sh"]
