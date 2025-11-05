#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

python manage.py collectstatic --no-input
python manage.py migrate

# Make start script executable
chmod +x start.sh

# Populate leagues if needed (uncomment if you want to run this on deployment)
# python manage.py populate_leagues


