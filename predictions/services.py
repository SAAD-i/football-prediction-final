import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from django.conf import settings

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not available. Install with: pip install onnxruntime")


# Mapping from URL slugs to folder names (for models_storage)
LEAGUE_FOLDER_MAP = {
    # Europe - Domestic Leagues
    'epl': 'EPL',
    'english-premier-league': 'EPL',
    'laliga-spain': 'LaLigaSpain',
    'italian-serie-a': 'SerieA',
    'german-bundesliga': 'BundesLiga',
    'french-ligue-1': 'Ligue1',
    'portuguese-primeira-liga': 'PremeiraLiga',
    'efl-championship': 'EFL',
    'scottish-premiership': 'ScotishPremiership',
    # Europe - Domestic Cups
    'english-fa-cup': 'English-FA-Cup',
    'english-carabao-cup': 'English-Carabao-Cup',
    'spanish-copa-del-rey': 'Spanish-Copa-del-Rey',
    'german-cup-dfb-pokal': 'German-Cup-DFB-Pokal',
    'coppa-italia': 'Coppa-Italia',
    'coupe-de-france': 'Coupe-de-France',
    'scottish-league-cup': 'Scottish-League-Cup',
}

# Mapping from URL slugs to training directory names (now matches models_storage folder names)
LEAGUE_TRAINING_FOLDER_MAP = {
    'epl': 'EPL',
    'english-premier-league': 'EPL',
    'laliga-spain': 'LaLigaSpain',  # Matches models_storage folder name
    'italian-serie-a': 'SerieA',
    'german-bundesliga': 'BundesLiga',
    'french-ligue-1': 'Ligue1',
    'portuguese-primeira-liga': 'PremeiraLiga',
    'efl-championship': 'EFL',
    'scottish-premiership': 'ScotishPremiership',
    # Europe - Domestic Cups
    'english-fa-cup': 'English-FA-Cup',
    'english-carabao-cup': 'English-Carabao-Cup',
    'spanish-copa-del-rey': 'Spanish-Copa-del-Rey',
    'german-cup-dfb-pokal': 'German-Cup-DFB-Pokal',
    'coppa-italia': 'Coppa-Italia',
    'coupe-de-france': 'Coupe-de-France',
    'scottish-league-cup': 'Scottish-League-Cup',
}

# Mapping from URL slugs to training script filenames
LEAGUE_TRAINING_SCRIPT_MAP = {
    'epl': 'train_epl_enhanced.py',
    'english-premier-league': 'train_epl_enhanced.py',
    'laliga-spain': 'train_laliga_enhanced.py',
    'italian-serie-a': None,  # Uses generic script
    'german-bundesliga': None,  # Uses generic script
    'french-ligue-1': None,  # Uses generic script
    'portuguese-primeira-liga': None,  # Uses generic script
    'efl-championship': None,  # Uses generic script
    'scottish-premiership': None,  # Uses generic script
}


def get_league_folder_name(league_slug: str) -> str:
    """Convert league slug to folder name"""
    normalized = league_slug.lower().strip()
    return LEAGUE_FOLDER_MAP.get(normalized, normalized)


class ONNXPredictor:
    """Service class for loading and using ONNX models for predictions"""
    
    def __init__(self, league_slug: str):
        self.league_slug = league_slug
        self.session = None
        self.class_names = ['H', 'D', 'A']  # Default order
        self.preprocessing_params = None
        
        # Load model for any league
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model for any league"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        # Get league folder name from slug
        folder_name = get_league_folder_name(self.league_slug)
        
        # Get model base path (now in Europe-Domestic-Leagues subfolder)
        model_base = Path(settings.BASE_DIR) / 'predictions' / 'models_storage' / 'Europe-Domestic-Leagues' / folder_name
        
        # Try different possible ONNX filenames
        league_slug_clean = self.league_slug.lower().replace('-', '_')
        onnx_filenames = [
            f'{folder_name.lower()}.onnx',  # e.g., bundesliga.onnx
            f'{league_slug_clean}.onnx',  # e.g., german_bundesliga.onnx
            'best_model_neural_network.onnx',
            'model.onnx',
        ]
        
        onnx_path = None
        
        # Try specific filenames first
        for filename in onnx_filenames:
            candidate_path = model_base / filename
            if candidate_path.exists():
                onnx_path = candidate_path
                break
        
        # If no specific file found, try finding any .onnx file
        if onnx_path is None:
            onnx_files = list(model_base.glob('*.onnx'))
            if onnx_files:
                onnx_path = onnx_files[0]  # Use first .onnx file found
        
        if onnx_path is None:
            raise FileNotFoundError(
                f"ONNX model not found in {model_base}. "
                f"Please place your .onnx model file in: {model_base}"
            )
        
        # Load ONNX model
        print(f"Loading ONNX model from: {onnx_path}")
        try:
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception:
            self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        expected_features = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        print(f"ONNX model loaded. Input: {self.input_name}, Shape: {input_shape}, Expected features: {expected_features}")
        
        # Log output information
        outputs_info = self.session.get_outputs()
        print(f"ONNX model outputs: {len(outputs_info)}")
        for i, output in enumerate(outputs_info):
            print(f"  Output {i}: name={output.name}, shape={output.shape}, type={output.type}")
        
        # Load preprocessing parameters from JSON
        json_path = model_base / 'preprocessing_parameters.json'
        if json_path.exists():
            print(f"Loading preprocessing parameters from: {json_path}")
            with open(json_path, 'r') as f:
                self.preprocessing_params = json.load(f)
            
            # Get class order from JSON
            if 'class_order' in self.preprocessing_params:
                self.class_names = self.preprocessing_params['class_order']
            print("Preprocessing parameters loaded successfully")
        else:
            self.preprocessing_params = None
            print(f"No preprocessing_parameters.json found in {model_base}")
        
        print(f"Model ready for league: {self.league_slug}")
        print(f"Class order: {self.class_names}")
        if self.session:
            input_shape = self.session.get_inputs()[0].shape
            print(f"Model input shape: {input_shape}")
    
    def _get_league_training_base(self):
        """Get training base path for this league"""
        if not hasattr(settings, 'TRAINING_BASE_DIR') or settings.TRAINING_BASE_DIR is None:
            return None
        
        normalized_slug = self.league_slug.lower().strip()
        training_folder = LEAGUE_TRAINING_FOLDER_MAP.get(normalized_slug)
        if not training_folder:
            return None
        
        training_path = Path(settings.TRAINING_BASE_DIR) / training_folder
        if training_path.exists():
            return training_path
        return None
    
    def _get_feature_row(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get feature DataFrame for prediction (returns DataFrame, not preprocessed array)"""
        # Import feature engineering functions from training directory
        import sys
        training_base = self._get_league_training_base()
        if training_base is None:
            raise ValueError(f"Training base not found for league: {self.league_slug}")
        
        print(f"DEBUG: Using training base: {training_base}")
        if str(training_base) not in sys.path:
            sys.path.insert(0, str(training_base))
        
        # Try to import league-specific training script first
        normalized_slug = self.league_slug.lower().strip()
        training_script = LEAGUE_TRAINING_SCRIPT_MAP.get(normalized_slug)
        
        engineer_features = None
        load_dataset = None
        DATA_PATH = None
        
        if training_script:
            # League-specific script (EPL, LaLiga)
            module_name = training_script.replace('.py', '')
            script_path = training_base / training_script
            print(f"DEBUG: Looking for training script: {script_path}")
            if script_path.exists():
                try:
                    training_module = __import__(module_name, fromlist=['engineer_features', 'load_dataset', 'DATA_PATH'])
                    engineer_features = training_module.engineer_features
                    load_dataset = training_module.load_dataset
                    DATA_PATH = training_module.DATA_PATH
                    print(f"DEBUG: Successfully imported {module_name}")
                except ImportError as e:
                    print(f"WARNING: Could not import {module_name}: {e}")
            else:
                print(f"WARNING: Training script not found: {script_path}")
        
        if engineer_features is None:
            # Generic script for other leagues (located in parent directory)
            import sys
            import importlib.util
            parent_dir = str(training_base.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Try importing from parent directory
            generic_script_path = Path(parent_dir) / 'train_league_generic.py'
            print(f"DEBUG: Looking for generic script: {generic_script_path}")
            if generic_script_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("train_league_generic", str(generic_script_path))
                    train_league_generic = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_league_generic)
                    engineer_features = train_league_generic.engineer_features
                    load_dataset = train_league_generic.load_dataset
                    print(f"DEBUG: Successfully imported train_league_generic")
                except Exception as e:
                    print(f"WARNING: Could not import train_league_generic: {e}")
            
            # Get data file path
            data_files = {
                'epl': 'epldata.csv',
                'english-premier-league': 'epldata.csv',
                'italian-serie-a': 'serieadata.csv',
                'german-bundesliga': 'bundesligadata.csv',
                'french-ligue-1': 'ligue1data.csv',
                'portuguese-primeira-liga': 'primeiraligadata.csv',
                'efl-championship': 'championshipdata.csv',
                'scottish-premiership': 'scottishpremdata.csv',
                'laliga-spain': 'laligadata.csv',
                # Europe - Domestic Cups
                'english-fa-cup': 'facupdata.csv',
                'english-carabao-cup': 'carabaocupdata.csv',
                'spanish-copa-del-rey': 'copadelreydata.csv',
                'german-cup-dfb-pokal': 'dfbpokaldata.csv',
                'coppa-italia': 'coppaitaliadata.csv',
                'coupe-de-france': 'coupedefrancedata.csv',
                'scottish-league-cup': 'scottishleaguecupdata.csv',
            }
            data_filename = data_files.get(normalized_slug)
            if not data_filename:
                raise ValueError(f"No data file mapping for league: {self.league_slug}")
            DATA_PATH = str(training_base / data_filename)
            print(f"DEBUG: Using data file: {DATA_PATH}")
        
        # If still no engineer_features function, raise error
        if engineer_features is None or load_dataset is None:
            raise FileNotFoundError(
                f"Training script not found for {self.league_slug}. "
                f"Expected: {training_script if training_script else 'train_league_generic.py'} "
                f"in {training_base} or {training_base.parent}"
            )
        
        # Load and engineer features (training data only, no 2024-25)
        if df is None:
            # Check if load_dataset supports include_2024_25 parameter
            try:
                df = load_dataset(DATA_PATH, include_2024_25=False)
            except TypeError:
                # Generic script doesn't have include_2024_25 parameter
                df = load_dataset(DATA_PATH)
        
        df_features = engineer_features(df)
        
        # Get most recent match stats for teams
        home_match = df_features[df_features["HomeTeam"] == home_team].tail(1)
        away_match = df_features[df_features["AwayTeam"] == away_team].tail(1)
        
        # Create feature DataFrame exactly like predict_match function
        if len(home_match) > 0 and len(away_match) > 0:
            home_feats = home_match.iloc[0]
            away_feats = away_match.iloc[0]
            
            return pd.DataFrame([{
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "home_form": home_feats["home_form"],
                "away_form": away_feats["away_form"],
                "home_goals_scored_avg": home_feats["home_goals_scored_avg"],
                "home_goals_conceded_avg": home_feats["home_goals_conceded_avg"],
                "away_goals_scored_avg": away_feats["away_goals_scored_avg"],
                "away_goals_conceded_avg": away_feats["away_goals_conceded_avg"],
                "h2h_home_win_rate": home_feats.get("h2h_home_win_rate", 0.33),
                "h2h_away_win_rate": home_feats.get("h2h_away_win_rate", 0.33),
                "h2h_draw_rate": home_feats.get("h2h_draw_rate", 0.33),
                "home_home_form": home_feats["home_home_form"],
                "away_away_form": away_feats["away_away_form"],
                "home_elo": home_feats["home_elo"],
                "away_elo": away_feats["away_elo"],
                "home_win_streak": home_feats["home_win_streak"],
                "away_win_streak": away_feats["away_win_streak"],
                "home_unbeaten_streak": home_feats["home_unbeaten_streak"],
                "away_unbeaten_streak": away_feats["away_unbeaten_streak"],
                "home_loss_streak": home_feats["home_loss_streak"],
                "away_loss_streak": away_feats["away_loss_streak"],
                "home_weighted_form": home_feats["home_weighted_form"],
                "away_weighted_form": away_feats["away_weighted_form"],
                "home_league_position": home_feats["home_league_position"],
                "away_league_position": away_feats["away_league_position"],
                "home_points": home_feats["home_points"],
                "away_points": away_feats["away_points"],
                "home_tier": home_feats["home_tier"],
                "away_tier": away_feats["away_tier"],
                "tier_diff": home_feats["tier_diff"],
                "tier_matchup": home_feats["tier_matchup"],
                "form_diff": home_feats["home_form"] - away_feats["away_form"],
                "goal_diff": (home_feats["home_goals_scored_avg"] - home_feats["home_goals_conceded_avg"]) - 
                            (away_feats["away_goals_scored_avg"] - away_feats["away_goals_conceded_avg"]),
                "home_advantage": home_feats["home_home_form"] - away_feats["away_away_form"],
                "elo_diff": home_feats["home_elo"] - away_feats["away_elo"],
                "position_diff": away_feats["away_league_position"] - home_feats["home_league_position"],
                "points_diff": home_feats["home_points"] - away_feats["away_points"],
                "weighted_form_diff": home_feats["home_weighted_form"] - away_feats["away_weighted_form"],
                "streak_momentum": (home_feats["home_win_streak"] - away_feats["away_win_streak"]) + 
                                  (away_feats["away_loss_streak"] - home_feats["home_loss_streak"]),
                "goal_expectation_home": home_feats["home_goals_scored_avg"] * (1 - away_feats["away_goals_conceded_avg"] / 3.0),
                "goal_expectation_away": away_feats["away_goals_scored_avg"] * (1 - home_feats["home_goals_conceded_avg"] / 3.0),
                "goal_expectation_diff": (home_feats["home_goals_scored_avg"] * (1 - away_feats["away_goals_conceded_avg"] / 3.0)) -
                                        (away_feats["away_goals_scored_avg"] * (1 - home_feats["home_goals_conceded_avg"] / 3.0)),
                "home_power": home_feats["home_power"],
                "away_power": away_feats["away_power"],
                "power_diff": home_feats["power_diff"],
                "elo_form_interaction": (home_feats["home_elo"] - away_feats["away_elo"]) * (home_feats["home_weighted_form"] - away_feats["away_weighted_form"]),
                "position_form_interaction": (away_feats["away_league_position"] - home_feats["home_league_position"]) * (home_feats["home_form"] - away_feats["away_form"]),
                "tier_form_interaction": home_feats["tier_form_interaction"]
            }])
        else:
            return pd.DataFrame([{
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "home_form": 1.5, "away_form": 1.5,
                "home_goals_scored_avg": 1.0, "home_goals_conceded_avg": 1.0,
                "away_goals_scored_avg": 1.0, "away_goals_conceded_avg": 1.0,
                "h2h_home_win_rate": 0.33, "h2h_away_win_rate": 0.33, "h2h_draw_rate": 0.33,
                "home_home_form": 1.5, "away_away_form": 1.5,
                "home_elo": 1500, "away_elo": 1500,
                "home_win_streak": 0, "away_win_streak": 0,
                "home_unbeaten_streak": 0, "away_unbeaten_streak": 0,
                "home_loss_streak": 0, "away_loss_streak": 0,
                "home_weighted_form": 1.5, "away_weighted_form": 1.5,
                "home_league_position": 10, "away_league_position": 10,
                "home_points": 0, "away_points": 0,
                "home_tier": 2, "away_tier": 2,
                "tier_diff": 0, "tier_matchup": 22,
                "form_diff": 0, "goal_diff": 0, "home_advantage": 0,
                "elo_diff": 0, "position_diff": 0, "points_diff": 0,
                "weighted_form_diff": 0, "streak_momentum": 0,
                "goal_expectation_home": 1.0, "goal_expectation_away": 1.0, "goal_expectation_diff": 0,
                "home_power": 25, "away_power": 25, "power_diff": 0,
                "elo_form_interaction": 0, "position_form_interaction": 0, "tier_form_interaction": 0
            }])
    
    def predict(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> Tuple[str, Dict[str, float]]:
        """Make prediction for a match using ONNX model"""
        if self.session is None:
            raise ValueError(f"Model not loaded for league: {self.league_slug}")
        
        # Always use ONNX model with preprocessing_parameters.json
        return self._predict_with_onnx(home_team, away_team, df)
    
    def _predict_with_onnx(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> Tuple[str, Dict[str, float]]:
        """Predict using ONNX model directly with preprocessing parameters - same approach as EPL"""
        # Prefer engineered features when training data is available; otherwise use safe defaults
        training_base = self._get_league_training_base()
        print(f"DEBUG: Training base for {self.league_slug}: {training_base}")
        if training_base is not None:
            try:
                print(f"DEBUG: Attempting to get feature row for {home_team} vs {away_team}")
                feature_row = self._get_feature_row(home_team, away_team, df)
                print(f"DEBUG: Successfully got feature row with {len(feature_row.columns)} columns")
            except Exception as e:
                import traceback
                print(f"ERROR: Engineered features unavailable for {home_team} vs {away_team}")
                print(f"ERROR: Exception: {str(e)}")
                print(f"ERROR: Traceback:")
                traceback.print_exc()
                print(f"WARNING: Falling back to default feature row.")
                feature_row = self._get_feature_row_simple(home_team, away_team)
        else:
            print(f"WARNING: Training base is None for {self.league_slug}, using default features")
            feature_row = self._get_feature_row_simple(home_team, away_team)
        
        # Get expected feature count from model
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic shapes (None or -1) - use a default if shape is not fully defined
        if isinstance(input_shape, (list, tuple)):
            # Filter out None and -1 (dynamic dimensions)
            shape_values = [s for s in input_shape if s is not None and s != -1]
            if len(shape_values) >= 2:
                expected_features = shape_values[1]
            elif len(shape_values) == 1:
                expected_features = shape_values[0]
            else:
                # Fallback: use preprocessing params if available
                if self.preprocessing_params and 'input_feature_count' in self.preprocessing_params:
                    expected_features = self.preprocessing_params['input_feature_count']
                else:
                    expected_features = 41  # Default EPL feature count
        else:
            # If shape is not a list/tuple, try to get feature count from preprocessing params
            if self.preprocessing_params and 'input_feature_count' in self.preprocessing_params:
                expected_features = self.preprocessing_params['input_feature_count']
            else:
                expected_features = 41  # Default EPL feature count
        
        print(f"DEBUG: Expected features: {expected_features}")
        
        # Extract team names for one-hot encoding
        home_team_name = feature_row['HomeTeam'].iloc[0] if 'HomeTeam' in feature_row.columns else home_team
        away_team_name = feature_row['AwayTeam'].iloc[0] if 'AwayTeam' in feature_row.columns else away_team
        
        # Get numeric features (exclude HomeTeam and AwayTeam)
        numeric_cols = feature_row.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = feature_row[numeric_cols].values.astype(np.float32)
        
        # One-hot encode teams if teams list is available in preprocessing_params
        team_features = None
        if self.preprocessing_params and 'teams' in self.preprocessing_params:
            teams_list = self.preprocessing_params['teams']
            print(f"DEBUG: One-hot encoding teams. Total teams: {len(teams_list)}")
            
            # Create one-hot encoding for HomeTeam and AwayTeam
            home_onehot = np.zeros(len(teams_list), dtype=np.float32)
            away_onehot = np.zeros(len(teams_list), dtype=np.float32)
            
            # Find team indices (case-insensitive matching)
            home_idx = None
            away_idx = None
            for i, team in enumerate(teams_list):
                if team.lower() == home_team_name.lower():
                    home_idx = i
                if team.lower() == away_team_name.lower():
                    away_idx = i
            
            if home_idx is not None:
                home_onehot[home_idx] = 1.0
                print(f"DEBUG: Home team '{home_team_name}' encoded at index {home_idx}")
            else:
                print(f"WARNING: Home team '{home_team_name}' not found in teams list. Available teams: {teams_list[:5]}...")
            
            if away_idx is not None:
                away_onehot[away_idx] = 1.0
                print(f"DEBUG: Away team '{away_team_name}' encoded at index {away_idx}")
            else:
                print(f"WARNING: Away team '{away_team_name}' not found in teams list. Available teams: {teams_list[:5]}...")
            
            # Combine team one-hot encodings: [home_onehot, away_onehot]
            team_features = np.concatenate([home_onehot, away_onehot]).reshape(1, -1)
            print(f"DEBUG: Team features shape: {team_features.shape}")
        else:
            print(f"WARNING: No teams list in preprocessing_params. Cannot one-hot encode teams.")
        
        # Ensure numeric_features is 2D: [1, num_features]
        if len(numeric_features.shape) == 1:
            numeric_features = numeric_features.reshape(1, -1)
        
        # Combine team features and numeric features
        if team_features is not None:
            feature_array = np.concatenate([team_features, numeric_features], axis=1)
            print(f"DEBUG: Combined features shape: {feature_array.shape} (team: {team_features.shape[1]}, numeric: {numeric_features.shape[1]})")
        else:
            feature_array = numeric_features
            print(f"WARNING: Using numeric features only (no team encoding). Shape: {feature_array.shape}")
        
        current_features = feature_array.shape[1]
        
        # Apply scaling from preprocessing_parameters.json if available
        # Note: scaler_mean/scale are for numeric features only (teams are already one-hot encoded)
        if self.preprocessing_params and 'scaler_mean' in self.preprocessing_params and 'scaler_scale' in self.preprocessing_params:
            # Ensure scaler_mean and scaler_scale are lists/arrays, not dicts
            scaler_mean_val = self.preprocessing_params['scaler_mean']
            scaler_scale_val = self.preprocessing_params['scaler_scale']
            
            if isinstance(scaler_mean_val, dict):
                raise ValueError("scaler_mean should be a list, not a dict. Check preprocessing_parameters.json")
            if isinstance(scaler_scale_val, dict):
                raise ValueError("scaler_scale should be a list, not a dict. Check preprocessing_parameters.json")
            
            scaler_mean = np.array(scaler_mean_val, dtype=np.float32)
            scaler_scale = np.array(scaler_scale_val, dtype=np.float32)
            
            # Ensure they're 1D arrays
            if len(scaler_mean.shape) > 1:
                scaler_mean = scaler_mean.flatten()
            if len(scaler_scale.shape) > 1:
                scaler_scale = scaler_scale.flatten()
            
            # Scale only numeric features (team features are already one-hot encoded, no scaling needed)
            if team_features is not None:
                # Feature array is [team_features, numeric_features]
                # Scale only the numeric part
                num_numeric = numeric_features.shape[1]
                if num_numeric == len(scaler_mean):
                    # Scale numeric features
                    scaled_numeric = (numeric_features - scaler_mean) / scaler_scale
                    # Recombine: [team_features, scaled_numeric_features]
                    feature_array = np.concatenate([team_features, scaled_numeric], axis=1)
                    current_features = feature_array.shape[1]
                    print(f"DEBUG: Scaled numeric features. Final feature array shape: {feature_array.shape}")
                else:
                    print(f"WARNING: Numeric feature count ({num_numeric}) doesn't match scaler ({len(scaler_mean)}). Padding/truncating.")
                    if num_numeric < len(scaler_mean):
                        padding = np.zeros((1, len(scaler_mean) - num_numeric), dtype=np.float32)
                        numeric_features = np.concatenate([numeric_features, padding], axis=1)
                    elif num_numeric > len(scaler_mean):
                        numeric_features = numeric_features[:, :len(scaler_mean)]
                    scaled_numeric = (numeric_features - scaler_mean) / scaler_scale
                    feature_array = np.concatenate([team_features, scaled_numeric], axis=1)
                    current_features = feature_array.shape[1]
            else:
                # No team features, scale all features
                if current_features == len(scaler_mean):
                    feature_array = (feature_array - scaler_mean) / scaler_scale
                else:
                    print(f"WARNING: Feature count ({current_features}) doesn't match scaler ({len(scaler_mean)}). Padding/truncating.")
                    if current_features < len(scaler_mean):
                        padding = np.zeros((1, len(scaler_mean) - current_features), dtype=np.float32)
                        feature_array = np.concatenate([feature_array, padding], axis=1)
                        current_features = feature_array.shape[1]
                    elif current_features > len(scaler_mean):
                        feature_array = feature_array[:, :len(scaler_mean)]
                        current_features = feature_array.shape[1]
                    feature_array = (feature_array - scaler_mean) / scaler_scale
        
        # Ensure feature count matches model input
        if current_features != expected_features:
            print(f"WARNING: Feature count mismatch. Current: {current_features}, Expected: {expected_features}")
            # Pad or truncate to match expected features
            if current_features < expected_features:
                padding = np.zeros((1, expected_features - current_features), dtype=np.float32)
                feature_array = np.concatenate([feature_array, padding], axis=1)
                print(f"DEBUG: Padded features to {expected_features}")
            elif current_features > expected_features:
                feature_array = feature_array[:, :expected_features]
                print(f"DEBUG: Truncated features to {expected_features}")
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: feature_array})
        
        # Debug: Print output shapes
        print(f"DEBUG: ONNX outputs count: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"DEBUG: Output {i} shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # Extract probabilities robustly: handle sklearn-onnx outputs (label + probability map)
        proba = None

        # 1) Handle list-of-dicts probability output: seq(map(int64, tensor(float)))
        for out in outputs:
            if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                prob_map = out[0]
                # Build probability vector by class index 0..2
                tmp = np.zeros(3, dtype=np.float32)
                for k, v in prob_map.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if idx < 0 or idx >= 3:
                        continue
                    # v may be scalar, 0-dim np array, or 1-element array
                    if isinstance(v, np.ndarray):
                        if v.size == 1:
                            tmp[idx] = float(v.ravel()[0])
                        else:
                            tmp[idx] = float(v[0])
                    else:
                        tmp[idx] = float(v)
                proba = tmp
                break

        # 2) If not found, fall back to numpy array heuristic
        if proba is None:
            candidate_arrays = []
            for out in outputs:
                if isinstance(out, np.ndarray):
                    arr = out
                    if arr.ndim == 2 and arr.shape[0] == 1:
                        arr = arr[0]
                    if arr.ndim == 1:
                        candidate_arrays.append(arr)

            float_candidates = [a for a in candidate_arrays if np.issubdtype(a.dtype, np.floating)]
            int_candidates = [a for a in candidate_arrays if np.issubdtype(a.dtype, np.integer)]

            if float_candidates:
                best = max(float_candidates, key=lambda a: a.shape[0])
                if best.shape[0] >= 3:
                    proba = best[:3]
                elif best.shape[0] == 2:
                    proba = np.array([best[0], best[1], max(0.0, 1.0 - float(best[0]) - float(best[1]))])
                elif best.shape[0] == 1:
                    v = float(best[0])
                    proba = np.array([v, 1.0 - v, 0.0])
            elif int_candidates:
                # Likely class index only; avoid degenerate one-hot 1/0/0 on Render
                idx_arr = int_candidates[0]
                pred_idx = int(idx_arr[0]) if idx_arr.shape[0] >= 1 else -1
                proba = np.array([0.33, 0.34, 0.33])
                if 0 <= pred_idx < 3:
                    proba[pred_idx] += 0.34
                    rest = (1.0 - proba[pred_idx]) / 2.0
                    for i in range(3):
                        if i != pred_idx:
                            proba[i] = rest

        if proba is None:
            print(f"WARNING: Could not extract probabilities, using defaults. Output types: {[type(o) for o in outputs]}")
            proba = np.array([0.33, 0.34, 0.33])
        
        # Ensure probabilities are valid numpy array with 3 values
        proba = np.array(proba).flatten()
        
        if len(proba) != 3:
            raise ValueError(f"Expected 3 probabilities, got {len(proba)}")
        
        # If we have logits (raw scores), convert to probabilities using softmax
        if proba.min() < 0 or proba.max() > 1.5:
            # Apply softmax to convert logits to probabilities
            exp_proba = np.exp(proba - np.max(proba))  # Numerical stability
            proba = exp_proba / exp_proba.sum()
        
        # Normalize probabilities (ensure they sum to 1)
        proba = proba / proba.sum() if proba.sum() > 0 else proba
        
        # Use class order from JSON or default
        classes = np.array(self.class_names)
        
        return self._format_prediction(proba, classes)
    
    def _get_feature_row_simple(self, home_team: str, away_team: str) -> pd.DataFrame:
        """Create feature DataFrame with default values (same as EPL's else clause)"""
        # Use the same default features as EPL when no training data is available
        return pd.DataFrame([{
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "home_form": 1.5, "away_form": 1.5,
            "home_goals_scored_avg": 1.0, "home_goals_conceded_avg": 1.0,
            "away_goals_scored_avg": 1.0, "away_goals_conceded_avg": 1.0,
            "h2h_home_win_rate": 0.33, "h2h_away_win_rate": 0.33, "h2h_draw_rate": 0.33,
            "home_home_form": 1.5, "away_away_form": 1.5,
            "home_elo": 1500, "away_elo": 1500,
            "home_win_streak": 0, "away_win_streak": 0,
            "home_unbeaten_streak": 0, "away_unbeaten_streak": 0,
            "home_loss_streak": 0, "away_loss_streak": 0,
            "home_weighted_form": 1.5, "away_weighted_form": 1.5,
            "home_league_position": 10, "away_league_position": 10,
            "home_points": 0, "away_points": 0,
            "home_tier": 2, "away_tier": 2,
            "tier_diff": 0, "tier_matchup": 22,
            "form_diff": 0, "goal_diff": 0, "home_advantage": 0,
            "elo_diff": 0, "position_diff": 0, "points_diff": 0,
            "weighted_form_diff": 0, "streak_momentum": 0,
            "goal_expectation_home": 1.0, "goal_expectation_away": 1.0, "goal_expectation_diff": 0,
            "home_power": 25, "away_power": 25, "power_diff": 0,
            "elo_form_interaction": 0, "position_form_interaction": 0, "tier_form_interaction": 0
        }])
    
    def _format_prediction(self, proba: np.ndarray, classes: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """Format prediction output"""
        # Ensure probabilities are valid
        proba = np.array(proba).flatten()
        if len(proba) != 3:
            raise ValueError(f"Expected 3 probabilities, got {len(proba)}")
        
        # Normalize probabilities
        proba = proba / proba.sum() if proba.sum() > 0 else proba
        
        # Apply draw-aware logic
        sorted_idx = np.argsort(proba)[::-1]
        top_prob = proba[sorted_idx[0]]
        second_prob = proba[sorted_idx[1]]
        
        # Draw-aware prediction: if top < 50% and difference < 15%, consider draw
        if top_prob < 0.50 and (top_prob - second_prob) < 0.15:
            draw_idx = np.where(classes == 'D')[0]
            if len(draw_idx) > 0 and proba[draw_idx[0]] > 0.20:
                predicted = 'D'
            else:
                predicted = classes[sorted_idx[0]]
        else:
            predicted = classes[sorted_idx[0]]
        
        # Map probabilities to standard H, D, A order
        h_idx = np.where(classes == 'H')[0][0] if 'H' in classes else 0
        d_idx = np.where(classes == 'D')[0][0] if 'D' in classes else 1
        a_idx = np.where(classes == 'A')[0][0] if 'A' in classes else 2
        
        probabilities = {
            'H': float(proba[h_idx]) if h_idx < len(proba) else 0.0,
            'D': float(proba[d_idx]) if d_idx < len(proba) else 0.0,
            'A': float(proba[a_idx]) if a_idx < len(proba) else 0.0,
        }
        
        # Debug output
        print(f"Debug - Class order: {classes.tolist()}")
        print(f"Debug - Probabilities: {proba}")
        print(f"Debug - Top prob: {top_prob:.3f}, Second: {second_prob:.3f}")
        print(f"Debug - Probabilities (mapped): H={probabilities['H']:.3f}, D={probabilities['D']:.3f}, A={probabilities['A']:.3f}")
        print(f"Debug - Predicted: {predicted}")
        
        return predicted, probabilities
