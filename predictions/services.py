import os
import json
import numpy as np
import pandas as pd
import joblib
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
}

# Mapping from URL slugs to training directory names (for Quick Delivery)
LEAGUE_TRAINING_FOLDER_MAP = {
    'epl': 'EPL',
    'english-premier-league': 'EPL',
    'laliga-spain': 'LaLiga-Spain',
    'italian-serie-a': 'Italian-Serie-A',
    'german-bundesliga': 'German-Bundesliga',
    'french-ligue-1': 'French-Ligue-1',
    'portuguese-primeira-liga': 'Portuguese-Primeira-Liga',
    'efl-championship': 'EFL-Championship',
    'scottish-premiership': 'Scottish-Premiership',
}

# Mapping from URL slugs to PKL model filenames
LEAGUE_PKL_MODEL_MAP = {
    'epl': 'best_model_neural_network.pkl',
    'english-premier-league': 'best_model_neural_network.pkl',
    'laliga-spain': 'best_model_laliga.pkl',
    'italian-serie-a': 'best_model_seriea.pkl',
    'german-bundesliga': 'best_model_bundesliga.pkl',
    'french-ligue-1': 'best_model_ligue1.pkl',
    'portuguese-primeira-liga': 'best_model_primeira.pkl',
    'efl-championship': 'best_model_championship.pkl',
    'scottish-premiership': 'best_model_scottish.pkl',
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
        self.preprocessor = None
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
        
        # Get model base path
        model_base = Path(settings.BASE_DIR) / 'predictions' / 'models_storage' / folder_name
        
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
        
        # Try to load PKL pipeline for all leagues (same as EPL approach)
        self._try_load_league_pipeline()
        
        print(f"Model ready for league: {self.league_slug}")
        print(f"Class order: {self.class_names}")
        if self.session:
            input_shape = self.session.get_inputs()[0].shape
            print(f"Model input shape: {input_shape}")
    
    def _try_load_league_pipeline(self):
        """Try to load PKL pipeline for any league (same as EPL approach)"""
        # Get training base path for this league
        training_base = self._get_league_training_base()
        if training_base is None:
            return
        
        # Get PKL filename for this league
        normalized_slug = self.league_slug.lower().strip()
        pkl_filename = LEAGUE_PKL_MODEL_MAP.get(normalized_slug)
        if not pkl_filename:
            return
        
        pkl_path = training_base / 'models' / pkl_filename
        
        if pkl_path.exists():
            try:
                print(f"Loading preprocessing pipeline from: {pkl_path}")
                saved_data = joblib.load(str(pkl_path))
                pipeline = saved_data["pipeline"]
                
                # Extract preprocessing pipeline
                base_estimator = pipeline
                if hasattr(pipeline, 'calibrated_classifiers_'):
                    if len(pipeline.calibrated_classifiers_) > 0:
                        base_estimator = pipeline.calibrated_classifiers_[0].estimator
                
                if hasattr(base_estimator, 'named_steps'):
                    self.preprocessor = base_estimator.named_steps.get("preprocess")
                    if self.preprocessor:
                        print("Preprocessing pipeline loaded successfully")
                        # Store pipeline for later use
                        self._pipeline = pipeline
            except Exception as e:
                print(f"Could not load PKL pipeline: {e}. Using JSON preprocessing.")
    
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
    
    def _preprocess_features(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> np.ndarray:
        """Preprocess features using the original pipeline preprocessor (like test_50.py)"""
        # Import feature engineering functions from training directory
        import sys
        training_base = self._get_league_training_base()
        if training_base is None:
            raise ValueError(f"Training base not found for league: {self.league_slug}")
        
        if str(training_base) not in sys.path:
            sys.path.insert(0, str(training_base))
        
        try:
            # Try to import league-specific training script first
            normalized_slug = self.league_slug.lower().strip()
            training_script = LEAGUE_TRAINING_SCRIPT_MAP.get(normalized_slug)
            
            if training_script:
                # League-specific script (EPL, LaLiga)
                module_name = training_script.replace('.py', '')
                training_module = __import__(module_name, fromlist=['engineer_features', 'load_dataset', 'DATA_PATH'])
                engineer_features = training_module.engineer_features
                load_dataset = training_module.load_dataset
                DATA_PATH = training_module.DATA_PATH
            else:
                # Generic script for other leagues (located in parent directory)
                import sys
                import importlib.util
                parent_dir = str(training_base.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                
                # Try importing from parent directory
                generic_script_path = Path(parent_dir) / 'train_league_generic.py'
                if generic_script_path.exists():
                    spec = importlib.util.spec_from_file_location("train_league_generic", str(generic_script_path))
                    train_league_generic = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(train_league_generic)
                    engineer_features = train_league_generic.engineer_features
                    load_dataset = train_league_generic.load_dataset
                else:
                    raise FileNotFoundError(f"train_league_generic.py not found at {generic_script_path}")
                
                # Get data file path
                data_files = {
                    'italian-serie-a': 'serieadata.csv',
                    'german-bundesliga': 'bundesligadata.csv',
                    'french-ligue-1': 'ligue1data.csv',
                    'portuguese-primeira-liga': 'primeiraligadata.csv',
                    'efl-championship': 'championshipdata.csv',
                    'scottish-premiership': 'scottishpremdata.csv',
                }
                data_filename = data_files.get(normalized_slug)
                if not data_filename:
                    raise ValueError(f"No data file mapping for league: {self.league_slug}")
                DATA_PATH = str(training_base / data_filename)
            
            # Load and engineer features (training data only, no 2024-25)
            if df is None:
                # Check if load_dataset supports include_2024_25 parameter
                try:
                    df = load_dataset(DATA_PATH, include_2024_25=False)
                except TypeError:
                    # Generic script doesn't have include_2024_25 parameter
                    df = load_dataset(DATA_PATH)
            
            df_features = engineer_features(df)
            
            # Get most recent match stats for teams (same as predict_match function)
            # Try to find most recent match with home team at home
            home_match = df_features[df_features["HomeTeam"] == home_team].tail(1)
            # Try to find most recent match with away team away
            away_match = df_features[df_features["AwayTeam"] == away_team].tail(1)
            
            # Create feature DataFrame exactly like predict_match function does
            if len(home_match) > 0 and len(away_match) > 0:
                # Extract features from home team's last home match
                home_feats = home_match.iloc[0]
                away_feats = away_match.iloc[0]
                
                # Create new prediction with combined features (same as predict_match)
                feature_row = pd.DataFrame([{
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
                # Use defaults if no recent data (same as predict_match function)
                feature_row = pd.DataFrame([{
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
            
            # Apply preprocessing using the original pipeline's preprocessor (like test_50.py)
            X_preprocessed = self.preprocessor.transform(feature_row).astype(np.float32)
            
            return X_preprocessed
            
        except Exception as e:
            print(f"Error in feature preprocessing: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _get_feature_row(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get feature DataFrame for prediction (returns DataFrame, not preprocessed array)"""
        # Import feature engineering functions from training directory
        import sys
        training_base = self._get_league_training_base()
        if training_base is None:
            raise ValueError(f"Training base not found for league: {self.league_slug}")
        
        if str(training_base) not in sys.path:
            sys.path.insert(0, str(training_base))
        
        # Try to import league-specific training script first
        normalized_slug = self.league_slug.lower().strip()
        training_script = LEAGUE_TRAINING_SCRIPT_MAP.get(normalized_slug)
        
        if training_script:
            # League-specific script (EPL, LaLiga)
            module_name = training_script.replace('.py', '')
            training_module = __import__(module_name, fromlist=['engineer_features', 'load_dataset', 'DATA_PATH'])
            engineer_features = training_module.engineer_features
            load_dataset = training_module.load_dataset
            DATA_PATH = training_module.DATA_PATH
        else:
            # Generic script for other leagues (located in parent directory)
            import sys
            import importlib.util
            parent_dir = str(training_base.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            # Try importing from parent directory
            generic_script_path = Path(parent_dir) / 'train_league_generic.py'
            if generic_script_path.exists():
                spec = importlib.util.spec_from_file_location("train_league_generic", str(generic_script_path))
                train_league_generic = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(train_league_generic)
                engineer_features = train_league_generic.engineer_features
                load_dataset = train_league_generic.load_dataset
            else:
                raise FileNotFoundError(f"train_league_generic.py not found at {generic_script_path}")
            
            # Get data file path
            data_files = {
                'italian-serie-a': 'serieadata.csv',
                'german-bundesliga': 'bundesligadata.csv',
                'french-ligue-1': 'ligue1data.csv',
                'portuguese-primeira-liga': 'primeiraligadata.csv',
                'efl-championship': 'championshipdata.csv',
                'scottish-premiership': 'scottishpremdata.csv',
            }
            data_filename = data_files.get(normalized_slug)
            if not data_filename:
                raise ValueError(f"No data file mapping for league: {self.league_slug}")
            DATA_PATH = str(training_base / data_filename)
        
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
        
        # For EPL with PKL pipeline, use the full pipeline approach
        if self.preprocessor is not None:
            return self._predict_with_pipeline(home_team, away_team, df)
        
        # For other leagues or when PKL is not available, use ONNX model directly
        return self._predict_with_onnx(home_team, away_team, df)
    
    def _predict_with_pipeline(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> Tuple[str, Dict[str, float]]:
        """Predict using PKL pipeline (same approach for all leagues)"""
        if not hasattr(self, '_pipeline'):
            training_base = self._get_league_training_base()
            if training_base is None:
                raise FileNotFoundError(f"Training base not found for league: {self.league_slug}")
        
            normalized_slug = self.league_slug.lower().strip()
            pkl_filename = LEAGUE_PKL_MODEL_MAP.get(normalized_slug)
            if not pkl_filename:
                raise FileNotFoundError(f"No PKL model mapping for league: {self.league_slug}")
            
            pkl_path = training_base / 'models' / pkl_filename
            if not pkl_path.exists():
                raise FileNotFoundError(f"PKL pipeline not found at {pkl_path}")
            saved_data = joblib.load(str(pkl_path))
            self._pipeline = saved_data["pipeline"]
        
        # Create feature DataFrame
        feature_row = self._get_feature_row(home_team, away_team, df)
        
        # Get probabilities from pipeline
        proba = self._pipeline.predict_proba(feature_row)[0]
        classes = self._pipeline.classes_ if hasattr(self._pipeline, 'classes_') else np.array(self.class_names)
        
        return self._format_prediction(proba, classes)
    
    def _predict_with_onnx(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> Tuple[str, Dict[str, float]]:
        """Predict using ONNX model directly with preprocessing parameters - same approach as EPL"""
        # Prefer engineered features when training data is available; otherwise use safe defaults
        training_base = self._get_league_training_base()
        if training_base is not None:
            try:
                feature_row = self._get_feature_row(home_team, away_team, df)
            except Exception as e:
                print(f"WARNING: Engineered features unavailable ({e}); using default feature row.")
                feature_row = self._get_feature_row_simple(home_team, away_team)
        else:
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
        
        # Convert DataFrame to numpy array (select numeric columns only)
        numeric_cols = feature_row.select_dtypes(include=[np.number]).columns
        feature_array = feature_row[numeric_cols].values.astype(np.float32)
        
        # Ensure feature_array is 2D: [1, num_features]
        if len(feature_array.shape) == 1:
            feature_array = feature_array.reshape(1, -1)
        
        current_features = feature_array.shape[1]
        
        # Apply scaling from preprocessing_parameters.json if available
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
            
            # Ensure we have the right number of features
            if current_features == len(scaler_mean):
                # Apply normalization: (x - mean) / scale
                feature_array = (feature_array - scaler_mean) / scaler_scale
            else:
                # If feature count doesn't match, pad or truncate
                if current_features < len(scaler_mean):
                    # Pad with zeros
                    padding = np.zeros((1, len(scaler_mean) - current_features), dtype=np.float32)
                    feature_array = np.concatenate([feature_array, padding], axis=1)
                    current_features = feature_array.shape[1]
                elif current_features > len(scaler_mean):
                    # Truncate
                    feature_array = feature_array[:, :len(scaler_mean)]
                    current_features = feature_array.shape[1]
                # Apply scaling
                feature_array = (feature_array - scaler_mean) / scaler_scale
        
        # Ensure feature count matches model input
        if current_features != expected_features:
            # Pad or truncate to match expected features
            if current_features < expected_features:
                padding = np.zeros((1, expected_features - current_features), dtype=np.float32)
                feature_array = np.concatenate([feature_array, padding], axis=1)
            elif current_features > expected_features:
                feature_array = feature_array[:, :expected_features]
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: feature_array})
        
        # Debug: Print output shapes
        print(f"DEBUG: ONNX outputs count: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"DEBUG: Output {i} shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        
        # Extract probabilities robustly: prefer float arrays with 3+ values; avoid class-index outputs
        proba = None
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
