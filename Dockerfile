# Bayesian Retrosynthesis - Docker Environment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 retrosynthesis && \
    chown -R retrosynthesis:retrosynthesis /app
USER retrosynthesis

# Expose port for potential web interface (optional)
EXPOSE 8000

# Default command - run the interactive demo
CMD ["python", "demo.py"]
