# AI Pipeline Production Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r pipeline && useradd -r -g pipeline pipeline

# Copy requirements and install Python dependencies
COPY requirements-production.txt .
RUN pip install --no-cache-dir -r requirements-production.txt

# Copy application code
COPY src/ ./src/
COPY main.py .
COPY README.md .

# Create necessary directories
RUN mkdir -p /app/data/logs /app/data/models /app/data/cache && \
    chown -R pipeline:pipeline /app

# Switch to non-root user
USER pipeline

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "src.deployment:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
