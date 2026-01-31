# Base image with Python 3.12
FROM python:3.12-slim

# Keep Python logs unbuffered and avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy dependency files first for faster builds
COPY pyproject.toml requirements.txt ./

# Install dependencies (without editable install yet)
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir --only-binary :all: -r requirements.txt

# Copy the rest of the project
COPY . .

# Install the project in editable mode (requires src/ to be present)
RUN python -m pip install -e .

# Default command (runs the live trading entrypoint)
CMD ["python", "-m", "src.main"]
