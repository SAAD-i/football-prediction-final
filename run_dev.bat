@echo off
REM Local development server startup script

echo Starting Django development server...
echo.

REM Run migrations if needed
echo Running migrations...
py manage.py migrate

REM Populate leagues if needed
echo Populating leagues...
py manage.py populate_leagues

echo.
echo Starting development server...
echo Server will be available at http://127.0.0.1:8000
echo Press Ctrl+C to stop the server
echo.

py manage.py runserver

