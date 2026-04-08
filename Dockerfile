# Use Astral's official uv image with Python 3.12 for extremely fast dependency installation
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install system dependencies necessary for some PyTorch operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation for performance
ENV UV_COMPILE_BYTECODE=1

# Tell uv where to place the environment
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen to ensure lockfile is strictly respected).
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Add current workspace to path
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

# Run the training script via accelerate
# This automatically scales across all available GPUs allocated to the container.
CMD ["accelerate", "launch", "scripts/run_experiment.py"]
