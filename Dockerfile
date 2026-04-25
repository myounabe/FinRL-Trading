# FinRL Trading Platform Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Fix: consolidate PYTHONPATH into a single ENV instruction to avoid the first one being overwritten
ENV PYTHONPATH=/app/src:/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    sqlite3 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY setup.py .
COPY README.md .
COPY .env* ./

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
# Also create a personal notebooks dir for my local experimentation
# Added models/ to persist trained model checkpoints between runs
# Added data/raw and data/processed subdirs to keep raw vs processed data organized
# Added experiments/ to track individual training runs with timestamped subdirs
# Added cache/ to store downloaded market data so I'm not re-fetching on every run
RUN mkdir -p data/raw data/processed logs notebooks models experiments cache

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import config; print('Health check passed')" || exit 1

# Expose port for web interface
EXPOSE 8501

# Default command - use the main CLI
CMD ["python", "src/main.py", "dashboard"]
