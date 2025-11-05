# Quick Setup Guide

## Step-by-Step Setup

1. **Navigate to Django directory:**
   ```powershell
   cd Django
   ```

2. **Activate virtual environment:**
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

3. **Create database migrations:**
   ```powershell
   python manage.py makemigrations
   ```

4. **Apply migrations (create database tables):**
   ```powershell
   python manage.py migrate
   ```

5. **Populate leagues and teams:**
   ```powershell
   python manage.py populate_leagues
   ```

6. **Run the development server:**
   ```powershell
   python manage.py runserver
   ```

## Access the Application

- **Homepage:** http://127.0.0.1:8000/
- **Admin Panel:** http://127.0.0.1:8000/admin/

## Troubleshooting

If you get "no such table" errors:
- Make sure you've run `makemigrations` and `migrate` first
- The migrations must be run BEFORE populating leagues

## Quick Setup Script

Alternatively, run the setup script:
```powershell
.\setup.bat
```

