#!/usr/bin/env bash
# Local development server startup script (for Linux/Mac)

echo "Starting Django development server..."
echo ""

# Run migrations if needed
echo "Running migrations..."
python manage.py migrate

# Populate leagues if needed
echo "Populating leagues..."
python manage.py populate_leagues

echo ""
echo "Starting development server..."
echo "Server will be available at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python manage.py runserver

