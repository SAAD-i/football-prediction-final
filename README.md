# Football Match Predictor - Django Project

A Django web application for predicting football match outcomes using machine learning ONNX models.

## Setup Instructions

1. **Activate Virtual Environment**
   ```powershell
   cd Django
   .\env\Scripts\Activate.ps1
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run Migrations**
   ```powershell
   python manage.py migrate
   ```

4. **Populate Leagues and Teams**
   ```powershell
   python manage.py populate_leagues
   ```

5. **Create Superuser (Optional)**
   ```powershell
   python manage.py createsuperuser
   ```

6. **Run Development Server**
   ```powershell
   python manage.py runserver
   ```

7. **Access the Application**
   - Web: http://127.0.0.1:8000/
   - Admin: http://127.0.0.1:8000/admin/

## Project Structure

```
Django/
├── football_predictor/          # Main project settings
├── predictions/                 # Predictions app
│   ├── models.py               # League and Team models
│   ├── views.py                # View functions
│   ├── services.py             # ONNX prediction service
│   └── management/commands/   # Management commands
├── templates/                  # HTML templates
│   ├── base.html
│   └── predictions/
│       ├── homepage.html
│       └── league_detail.html
└── requirements.txt            # Python dependencies
```

## Features

- **Homepage**: Display all football leagues organized by category
- **League Detail**: View teams and make predictions for a specific league
- **EPL Prediction**: Fully functional prediction using ONNX model
- **Modern UI**: Beautiful TailwindCSS-styled interface

## Currently Supported Leagues

- ✅ English Premier League (EPL) - Full prediction support
- ⏳ Other leagues - Coming soon (structure ready)

## ONNX Model Integration

The application uses ONNX Runtime to load and run quantized neural network models. For EPL, it uses:
- Model: `best_model_neural_network_quantized.onnx`
- Preprocessing: Uses the original sklearn pipeline for feature engineering

## Notes

- Ensure the EPL model files are in the correct path: `../Quick Delivery/Europe-Domestic-Leagues/EPL/models/`
- The application requires the original dataset CSV for feature engineering
- For production, consider caching the ONNX model sessions

