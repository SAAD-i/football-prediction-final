#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Make start script executable
chmod +x start.sh

# Run migrations (creates database tables)
echo "Running migrations..."
python manage.py migrate --no-input

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --no-input

# Populate leagues and teams (required for homepage to work)
echo "Populating leagues..."
python manage.py populate_leagues || echo "Warning: populate_leagues failed, but continuing..."


