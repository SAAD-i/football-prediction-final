#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate

# Make start script executable
chmod +x start.sh

# Populate leagues and teams (required for homepage to work)
python manage.py populate_leagues


