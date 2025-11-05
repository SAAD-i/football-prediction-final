# Render Deployment Guide

## Quick Setup

1. **Connect your repository** to Render
2. **Configure the service** with these settings:

### Build Command:
```bash
pip install --upgrade pip && pip install -r requirements.txt && python manage.py collectstatic --no-input && python manage.py migrate
```

### Start Command:
You have two options:

**Option 1: Use the start script (Recommended)**
```bash
./start.sh
```

**Option 2: Direct gunicorn command**
```bash
gunicorn football_predictor.wsgi:application --bind 0.0.0.0:$PORT
```

**Important**: 
- **Do NOT use** `python manage.py runserver` - it doesn't bind to `0.0.0.0` properly
- Make sure to use `gunicorn` and bind to `0.0.0.0:$PORT` 
- The `PORT` environment variable is automatically provided by Render (default: `10000`)

### Environment Variables:

Set these in your Render dashboard under **Environment**:

- `PORT`: **Automatically provided by Render** (default: `10000`). 
  - Render automatically sets this environment variable
  - You can override it in Render dashboard: Go to your service → **Environment** → Add `PORT` with your desired value (e.g., `10000`)
  - Usually not necessary to change, but useful if you need a specific port
  - The start script (`start.sh`) reads this variable and uses it automatically
- `DEBUG`: Set to `False` for production
- `SECRET_KEY`: **Required** - Set a secure secret key. Generate one using:
  ```python
  python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
  ```
- `ALLOWED_HOSTS`: (Optional) Comma-separated list of additional hosts beyond your Render domain

## Alternative: Using render.yaml

If you've added `render.yaml` to your repository, Render will automatically detect and use these settings.

## Important Notes:

1. **Port Binding**: 
   - Render automatically provides the `PORT` environment variable (default: `10000`)
   - The start command uses `gunicorn` with `--bind 0.0.0.0:$PORT` to bind to all interfaces
   - **Do NOT use** `python manage.py runserver` as it doesn't bind to `0.0.0.0` properly
   - You can override PORT in Render dashboard if needed, but it's usually not necessary

2. **Static Files**: Make sure `STATIC_ROOT` is set correctly (already configured in settings.py)

3. **Database**: Currently using SQLite. For production, consider using PostgreSQL:
   - Add `psycopg2-binary` to requirements.txt
   - Configure DATABASES in settings.py to use Render's PostgreSQL database

4. **Secret Key**: Generate a secure secret key and set it as an environment variable in Render dashboard

5. **Training Directory (Required for Predictions)**: 
   - The EPL prediction feature requires access to the training directory with:
     - `models/best_model_neural_network.pkl` (preprocessing pipeline)
     - `train_epl_enhanced.py` (feature engineering functions)
     - `epldata.csv` (historical match data)
   - Currently, the app expects this at: `Quick Delivery/Europe-Domestic-Leagues/EPL/`
   - **This path won't exist on Render by default**
   - **Solutions**:
     - Option A: Copy the necessary files to your Django app directory (recommended)
     - Option B: Upload files to a cloud storage service and download during build
     - Option C: Include the training directory in your repository (if not too large)
   - Without these files, the homepage will load but predictions will fail

## Troubleshooting Common Issues

### "No leagues available" / "no such table" Error

If you see "No leagues available" or `sqlite3.OperationalError: no such table: predictions_league`:

**This means migrations haven't run successfully. Fix it:**

**On Render:**
1. Go to your service dashboard
2. Click on **Shell** tab (or use **One-off Jobs**)
3. Run these commands in order:
   ```bash
   python manage.py migrate
   python manage.py populate_leagues
   ```
4. Refresh your website

**Locally:**
```bash
py manage.py migrate
py manage.py populate_leagues
```

**Note:** The `build.sh` script should run migrations automatically, but if it fails, you need to run them manually.

### 500 Errors

If you're seeing a 500 error:

1. **Check logs** in Render dashboard for the specific error
2. **Database issues**: Ensure migrations ran successfully (`python manage.py migrate`)
3. **Missing data**: Ensure `populate_leagues` command ran (already in build.sh)
4. **Training directory**: If predictions fail, check if EPL_TRAINING_BASE path exists
5. **Static files**: Ensure `collectstatic` ran successfully

## Generating a Secret Key

Run this command to generate a secure secret key:
```python
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Copy the output and add it as the `SECRET_KEY` environment variable in Render.

