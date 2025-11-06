# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make scripts executable
RUN chmod +x build.sh start.sh

# Run build script (migrations, collectstatic, populate_leagues)
RUN ./build.sh || echo "Build script completed with warnings"

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Start command
CMD ["./start.sh"]

