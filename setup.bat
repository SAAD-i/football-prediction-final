@echo off
REM Setup script for Django Football Predictor

echo Activating virtual environment...
call env\Scripts\activate.bat

echo.
echo Creating migrations...
python manage.py makemigrations

echo.
echo Running migrations...
python manage.py migrate

echo.
echo Populating leagues...
python manage.py populate_leagues

echo.
echo Setup complete! You can now run the server with:
echo   python manage.py runserver
echo.
pause

