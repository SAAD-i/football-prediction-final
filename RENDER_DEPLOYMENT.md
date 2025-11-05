# Render Deployment Guide

## Quick Setup

1. **Connect your repository** to Render
2. **Configure the service** with these settings:

### Build Command:
```bash
pip install --upgrade pip && pip install -r requirements.txt && python manage.py collectstatic --no-input && python manage.py migrate
```

### Start Command:
```bash
gunicorn football_predictor.wsgi:application --bind 0.0.0.0:$PORT
```

### Environment Variables:
- `DEBUG`: Set to `False` for production
- `SECRET_KEY`: (Optional) Set a secure secret key. If not set, it will use the default (not recommended for production)
- `ALLOWED_HOSTS`: (Optional) Comma-separated list of additional hosts

## Alternative: Using render.yaml

If you've added `render.yaml` to your repository, Render will automatically detect and use these settings.

## Important Notes:

1. **Static Files**: Make sure `STATIC_ROOT` is set correctly (already configured in settings.py)
2. **Database**: Currently using SQLite. For production, consider using PostgreSQL:
   - Add `psycopg2-binary` to requirements.txt
   - Configure DATABASES in settings.py to use Render's PostgreSQL database
3. **Secret Key**: Generate a secure secret key and set it as an environment variable in Render dashboard
4. **Port Binding**: The start command uses `$PORT` which Render automatically provides

## Generating a Secret Key

Run this command to generate a secure secret key:
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Copy the output and add it as the `SECRET_KEY` environment variable in Render.

