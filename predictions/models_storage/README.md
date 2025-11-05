# Model Storage Guide for Django App

## Where to Place ONNX Models

You have two options for storing models:

### Option 1: External Directory (Current Setup)
Models are stored in the training directory:
```
Quick Delivery/
└── Europe-Domestic-Leagues/
    └── EPL/
        └── models/
            ├── best_model_neural_network_quantized.onnx
            ├── best_model_neural_network.onnx
            └── best_model_neural_network.pkl
```

**Pros:** 
- Keeps models with training code
- Easy to access during development

**Cons:**
- Paths depend on directory structure
- May not work well in production

### Option 2: Django App Directory (Recommended for Production)
Store models within the Django app:
```
Django/
└── predictions/
    └── models_storage/
        ├── EPL/
        │   └── models/
        │       ├── best_model_neural_network_quantized.onnx
        │       ├── best_model_neural_network.onnx
        │       └── best_model_neural_network.pkl
        ├── LaLiga/
        │   └── models/
        │       └── ...
        └── SerieA/
            └── models/
                └── ...
```

**Pros:**
- Self-contained Django app
- Better for deployment
- Version control friendly

**Cons:**
- Models stored in repository (may be large)

## Quick Setup for Django App Directory

1. **Create the directory structure:**
   ```
   Django/predictions/models_storage/EPL/models/
   ```

2. **Copy your EPL model files:**
   - `best_model_neural_network_quantized.onnx`
   - `best_model_neural_network.onnx` (optional)
   - `best_model_neural_network.pkl`
   - `preprocessing_parameters.json` (if available)

3. **Update settings.py:**
   Uncomment the line in settings.py:
   ```python
   EPL_MODEL_BASE = BASE_DIR / 'predictions' / 'models_storage' / 'EPL'
   ```

4. **Add to .gitignore (if models are large):**
   ```
   Django/predictions/models_storage/*/*.onnx
   Django/predictions/models_storage/*/*.pkl
   ```

## File Structure Template

```
Django/
└── predictions/
    └── models_storage/
        ├── EPL/
        │   ├── models/
        │   │   ├── best_model_neural_network_quantized.onnx ✅
        │   │   ├── best_model_neural_network.onnx ✅
        │   │   └── best_model_neural_network.pkl ✅
        │   └── preprocessing_parameters.json (optional)
        ├── LaLiga-Spain/
        │   └── models/
        │       └── ...
        └── README.md (this file)
```

## Current Configuration

Currently using: **Option 1 (External Directory)**

To switch to Django app directory, update `settings.py` and move/copy your models.

