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


class ONNXPredictor:
    """Service class for loading and using ONNX models for predictions"""
    
    def __init__(self, league_slug: str):
        self.league_slug = league_slug
        self.session = None
        self.preprocessor = None
        self.class_names = ['H', 'D', 'A']  # Default order
        
        # Normalize league slug
        if league_slug in ['epl', 'english-premier-league']:
            self._load_epl_model()
    
    def _load_epl_model(self):
        """Load EPL ONNX model and preprocessing pipeline (like test_50.py)"""
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")
        
        # Paths to EPL ONNX model (from Django app directory) - use non-quantized model
        epl_model_base = Path(settings.EPL_MODEL_BASE)
        
        # Try different possible ONNX filenames (prioritize non-quantized)
        onnx_filenames = [
            'best_model_neural_network.onnx',  # Non-quantized model
            'epl.onnx', 
            'EPL.onnx', 
            'best_model_neural_network_quantized.onnx'
        ]
        onnx_path = None
        
        for filename in onnx_filenames:
            candidate_path = epl_model_base / filename
            if candidate_path.exists():
                onnx_path = candidate_path
                break
        
        if onnx_path is None:
            raise FileNotFoundError(f"ONNX model not found in {epl_model_base}. Tried: {onnx_filenames}")
        
        # Load ONNX model
        print(f"Loading ONNX model from: {onnx_path}")
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        expected_features = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        print(f"ONNX model loaded. Input: {self.input_name}, Shape: {input_shape}, Expected features: {expected_features}")
        
        # Load original pipeline for preprocessing (like test_50.py does)
        epl_training_base = Path(settings.EPL_TRAINING_BASE)
        pkl_path = epl_training_base / 'models' / 'best_model_neural_network.pkl'
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"Original model not found at {pkl_path}. Need it for preprocessing pipeline.")
        
        print(f"Loading preprocessing pipeline from: {pkl_path}")
        try:
            saved_data = joblib.load(str(pkl_path))
        except ModuleNotFoundError as e:
            if 'xgboost' in str(e).lower():
                raise ImportError(
                    f"XGBoost is required to load the preprocessing pipeline. "
                    f"Please install it with: pip install xgboost\n"
                    f"Error: {e}"
                )
            raise
        pipeline = saved_data["pipeline"]
        train_metrics = saved_data.get("metrics", {})
        
        # Extract preprocessing pipeline (same as test_50.py)
        base_estimator = pipeline
        if hasattr(pipeline, 'calibrated_classifiers_'):
            if len(pipeline.calibrated_classifiers_) > 0:
                base_estimator = pipeline.calibrated_classifiers_[0].estimator
        
        if not hasattr(base_estimator, 'named_steps'):
            raise ValueError("Could not extract preprocessing pipeline")
        
        self.preprocessor = base_estimator.named_steps.get("preprocess")
        if self.preprocessor is None:
            raise ValueError("Preprocessor not found in pipeline")
        
        # Get class order from the model itself (same as predict_match function)
        if hasattr(pipeline, 'classes_'):
            self.class_names = pipeline.classes_.tolist()
        elif hasattr(base_estimator, 'named_steps'):
            model_step = base_estimator.named_steps.get('model')
            if model_step and hasattr(model_step, 'classes_'):
                self.class_names = model_step.classes_.tolist()
            else:
                self.class_names = train_metrics.get('classes_order', ['H', 'D', 'A'])
        else:
            self.class_names = train_metrics.get('classes_order', ['H', 'D', 'A'])
        
        print(f"Preprocessing pipeline loaded successfully")
        print(f"Class order: {self.class_names}")
        print(f"Expected input features: {expected_features}")
    
    def _preprocess_features(self, home_team: str, away_team: str, df: pd.DataFrame = None) -> np.ndarray:
        """Preprocess features using the original pipeline preprocessor (like test_50.py)"""
        # Import feature engineering functions from training directory
        import sys
        epl_training_path = Path(settings.EPL_TRAINING_BASE)
        if str(epl_training_path) not in sys.path:
            sys.path.insert(0, str(epl_training_path))
        
        try:
            from train_epl_enhanced import engineer_features, load_dataset, DATA_PATH
            
            # Load and engineer features (training data only, no 2024-25)
            if df is None:
                df = load_dataset(DATA_PATH, include_2024_25=False)
            
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
        epl_training_path = Path(settings.EPL_TRAINING_BASE)
        if str(epl_training_path) not in sys.path:
            sys.path.insert(0, str(epl_training_path))
        
        from train_epl_enhanced import engineer_features, load_dataset, DATA_PATH
        
        # Load and engineer features (training data only, no 2024-25)
        if df is None:
            df = load_dataset(DATA_PATH, include_2024_25=False)
        
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
        """Make prediction for a match (following test_50.py method)"""
        if self.session is None:
            raise ValueError(f"Model not loaded for league: {self.league_slug}")
        
        # Get probabilities from original pipeline (like predict_match function)
        # The ONNX model only has the neural network component, so we use the full pipeline for probabilities
        # Load pipeline once and cache it
        if not hasattr(self, '_pipeline'):
            epl_training_base = Path(settings.EPL_TRAINING_BASE)
            pkl_path = epl_training_base / 'models' / 'best_model_neural_network.pkl'
            saved_data = joblib.load(str(pkl_path))
            self._pipeline = saved_data["pipeline"]
        
        # Create feature DataFrame (reuse logic from _preprocess_features but get DataFrame)
        feature_row = self._get_feature_row(home_team, away_team, df)
        
        # Get probabilities from pipeline (same as predict_match)
        proba = self._pipeline.predict_proba(feature_row)[0]
        classes = self._pipeline.classes_ if hasattr(self._pipeline, 'classes_') else np.array(self.class_names)
        
        # Ensure probabilities are valid (same as predict_match function)
        proba = np.array(proba)
        if len(proba) != 3:
            raise ValueError(f"Expected 3 probabilities, got {len(proba)}")
        
        # Normalize probabilities
        proba = proba / proba.sum() if proba.sum() > 0 else proba
        
        # Apply draw-aware logic (same as predict_match function)
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
