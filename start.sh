#!/usr/bin/env bash
# Start script for Render deployment
# This script ensures migrations run and gunicorn binds to the PORT environment variable

# Get PORT from environment variable (default to 10000 if not set)
PORT=${PORT:-8000}

# Run migrations if database tables don't exist
echo "Checking database..."
python manage.py migrate --no-input || echo "Warning: Migrations failed, but continuing..."

# Populate leagues if they don't exist
echo "Populating leagues..."
python manage.py populate_leagues || echo "Warning: populate_leagues failed, but continuing..."

# Start gunicorn with proper binding
echo "Starting gunicorn on port $PORT..."
exec gunicorn football_predictor.wsgi:application --bind 0.0.0.0:$PORT


