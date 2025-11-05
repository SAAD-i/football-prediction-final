#!/usr/bin/env bash
# Start script for Render deployment
# This script ensures gunicorn binds to the PORT environment variable

# Get PORT from environment variable (default to 10000 if not set)
PORT=${PORT:-10000}

# Start gunicorn with proper binding
exec gunicorn football_predictor.wsgi:application --bind 0.0.0.0:$PORT

