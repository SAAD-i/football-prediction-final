"""
Generic training script for all leagues
Usage: python train_league_generic.py --league "EFL-Championship" --data-file "championshipdata.csv"
"""
import argparse
import os
import sys
import warnings
from typing import Tuple, List, Dict
from collections import defaultdict
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')

# Import all the feature engineering functions from EPL training script
# These are identical for all leagues
def calculate_team_form(df: pd.DataFrame, window: int = 5) -> Dict:
    team_form = defaultdict(lambda: defaultdict(float))
    team_matches = defaultdict(list)
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        
        home_recent = team_matches[home][-window:]
        away_recent = team_matches[away][-window:]
        
        home_form = sum(home_recent) / len(home_recent) if home_recent else 1.5
        away_form = sum(away_recent) / len(away_recent) if away_recent else 1.5
        
        team_form[home][date] = home_form
        team_form[away][date] = away_form
        
        if result == "H":
            team_matches[home].append(3)
            team_matches[away].append(0)
        elif result == "A":
            team_matches[home].append(0)
            team_matches[away].append(3)
        else:
            team_matches[home].append(1)
            team_matches[away].append(1)
    
    return team_form


def calculate_goal_stats(df: pd.DataFrame, window: int = 5) -> Dict:
    team_goals_scored = defaultdict(list)
    team_goals_conceded = defaultdict(list)
    goal_stats = defaultdict(lambda: defaultdict(lambda: {"scored": 0.0, "conceded": 0.0}))
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        home_goals = row["FTHG"]
        away_goals = row["FTAG"]
        
        home_scored_recent = team_goals_scored[home][-window:]
        home_conceded_recent = team_goals_conceded[home][-window:]
        away_scored_recent = team_goals_scored[away][-window:]
        away_conceded_recent = team_goals_conceded[away][-window:]
        
        goal_stats[home][date]["scored"] = np.mean(home_scored_recent) if home_scored_recent else 1.0
        goal_stats[home][date]["conceded"] = np.mean(home_conceded_recent) if home_conceded_recent else 1.0
        goal_stats[away][date]["scored"] = np.mean(away_scored_recent) if away_scored_recent else 1.0
        goal_stats[away][date]["conceded"] = np.mean(away_conceded_recent) if away_conceded_recent else 1.0
        
        team_goals_scored[home].append(home_goals)
        team_goals_conceded[home].append(away_goals)
        team_goals_scored[away].append(away_goals)
        team_goals_conceded[away].append(home_goals)
    
    return goal_stats


def calculate_head_to_head(df: pd.DataFrame) -> Dict:
    h2h = defaultdict(lambda: {"home_wins": 0, "draws": 0, "away_wins": 0, "total": 0})
    
    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        key = (home, away)
        
        if result == "H":
            h2h[key]["home_wins"] += 1
        elif result == "A":
            h2h[key]["away_wins"] += 1
        else:
            h2h[key]["draws"] += 1
        h2h[key]["total"] += 1
    
    return h2h


def calculate_home_away_stats(df: pd.DataFrame, window: int = 10) -> Dict:
    home_stats = defaultdict(list)
    away_stats = defaultdict(list)
    team_home_away = defaultdict(lambda: defaultdict(lambda: {"home_form": 1.5, "away_form": 1.5}))
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        
        home_recent = home_stats[home][-window:]
        away_recent = away_stats[away][-window:]
        
        home_form = sum(home_recent) / len(home_recent) if home_recent else 1.5
        away_form = sum(away_recent) / len(away_recent) if away_recent else 1.5
        
        team_home_away[home][date]["home_form"] = home_form
        team_home_away[away][date]["away_form"] = away_form
        
        if result == "H":
            home_stats[home].append(3)
            away_stats[away].append(0)
        elif result == "A":
            home_stats[home].append(0)
            away_stats[away].append(3)
        else:
            home_stats[home].append(1)
            away_stats[away].append(1)
    
    return team_home_away


def calculate_elo_ratings(df: pd.DataFrame, k_factor: float = 40.0, initial_rating: float = 1500.0) -> Dict:
    team_elo = defaultdict(lambda: initial_rating)
    elo_history = defaultdict(lambda: defaultdict(float))
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        home_goals = row["FTHG"]
        away_goals = row["FTAG"]
        
        home_elo = team_elo[home]
        away_elo = team_elo[away]
        
        elo_history[home][date] = home_elo
        elo_history[away][date] = away_elo
        
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - 100) / 400))
        expected_away = 1 - expected_home
        
        if result == "H":
            actual_home, actual_away = 1.0, 0.0
        elif result == "A":
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5
        
        goal_diff = abs(home_goals - away_goals)
        multiplier = math.log(max(goal_diff, 1) + 1)
        
        team_elo[home] += k_factor * multiplier * (actual_home - expected_home)
        team_elo[away] += k_factor * multiplier * (actual_away - expected_away)
    
    return elo_history


def calculate_win_streaks(df: pd.DataFrame) -> Dict:
    team_results = defaultdict(list)
    streak_history = defaultdict(lambda: defaultdict(lambda: {"win_streak": 0, "unbeaten_streak": 0, "loss_streak": 0}))
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        
        for team in [home, away]:
            recent = team_results[team][-10:] if team_results[team] else []
            
            win_streak = 0
            for r in reversed(recent):
                if r == "W":
                    win_streak += 1
                else:
                    break
            
            unbeaten_streak = 0
            for r in reversed(recent):
                if r in ["W", "D"]:
                    unbeaten_streak += 1
                else:
                    break
            
            loss_streak = 0
            for r in reversed(recent):
                if r == "L":
                    loss_streak += 1
                else:
                    break
            
            streak_history[team][date] = {
                "win_streak": win_streak,
                "unbeaten_streak": unbeaten_streak,
                "loss_streak": loss_streak
            }
        
        if result == "H":
            team_results[home].append("W")
            team_results[away].append("L")
        elif result == "A":
            team_results[home].append("L")
            team_results[away].append("W")
        else:
            team_results[home].append("D")
            team_results[away].append("D")
    
    return streak_history


def calculate_weighted_form(df: pd.DataFrame, window: int = 10, decay: float = 0.85, super_recent: int = 5) -> Dict:
    team_matches = defaultdict(list)
    weighted_form = defaultdict(lambda: defaultdict(float))
    
    for idx, row in df.iterrows():
        date = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        
        for team in [home, away]:
            recent = team_matches[team][-window:]
            if recent:
                weights = []
                for i in range(len(recent), 0, -1):
                    base_weight = decay ** i
                    if i <= super_recent:
                        weights.append(base_weight * 10.0)
                    else:
                        weights.append(base_weight)
                
                weighted_avg = sum(r * w for r, w in zip(recent, weights)) / sum(weights)
                weighted_form[team][date] = weighted_avg
            else:
                weighted_form[team][date] = 1.5
        
        if result == "H":
            team_matches[home].append(3)
            team_matches[away].append(0)
        elif result == "A":
            team_matches[home].append(0)
            team_matches[away].append(3)
        else:
            team_matches[home].append(1)
            team_matches[away].append(1)
    
    return weighted_form


def calculate_team_tiers(df: pd.DataFrame) -> Dict:
    recent_cutoff = df['Date'].max() - pd.Timedelta(days=730)
    recent_df = df[df['Date'] >= recent_cutoff]
    
    team_points = defaultdict(lambda: {"points": 0, "matches": 0})
    
    for idx, row in recent_df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]
        result = row["FTR"]
        
        team_points[home]["matches"] += 1
        team_points[away]["matches"] += 1
        
        if result == "H":
            team_points[home]["points"] += 3
        elif result == "A":
            team_points[away]["points"] += 3
        else:
            team_points[home]["points"] += 1
            team_points[away]["points"] += 1
    
    team_ppm = {}
    for team, stats in team_points.items():
        if stats["matches"] > 10:
            team_ppm[team] = stats["points"] / stats["matches"]
        else:
            team_ppm[team] = 1.0
    
    sorted_teams = sorted(team_ppm.items(), key=lambda x: x[1], reverse=True)
    
    team_tiers = {}
    num_teams = len(sorted_teams)
    
    for i, (team, ppm) in enumerate(sorted_teams):
        if i < num_teams * 0.30:
            team_tiers[team] = 1
        elif i < num_teams * 0.70:
            team_tiers[team] = 2
        else:
            team_tiers[team] = 3
    
    return team_tiers


def calculate_league_position_stats(df: pd.DataFrame) -> Dict:
    df['Season'] = df['Date'].dt.year
    
    team_stats = defaultdict(lambda: defaultdict(lambda: {"points": 0, "played": 0, "position": 10}))
    
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season].sort_values('Date')
        season_points = defaultdict(int)
        season_played = defaultdict(int)
        
        for idx, row in season_df.iterrows():
            date = row["Date"]
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            result = row["FTR"]
            
            all_teams = list(set(list(season_points.keys())))
            if all_teams:
                sorted_teams = sorted(all_teams, key=lambda t: (-season_points[t], t))
                for pos, team in enumerate(sorted_teams, 1):
                    if team == home:
                        team_stats[home][date] = {
                            "points": season_points[home],
                            "played": season_played[home],
                            "position": pos
                        }
                    if team == away:
                        team_stats[away][date] = {
                            "points": season_points[away],
                            "played": season_played[away],
                            "position": pos
                        }
            
            season_played[home] += 1
            season_played[away] += 1
            
            if result == "H":
                season_points[home] += 3
            elif result == "A":
                season_points[away] += 3
            else:
                season_points[home] += 1
                season_points[away] += 1
    
    return team_stats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to the dataset"""
    print("Engineering features...")
    
    team_form = calculate_team_form(df, window=5)
    goal_stats = calculate_goal_stats(df, window=5)
    h2h = calculate_head_to_head(df)
    home_away_stats = calculate_home_away_stats(df, window=10)
    elo_ratings = calculate_elo_ratings(df, k_factor=40.0)
    streaks = calculate_win_streaks(df)
    weighted_form = calculate_weighted_form(df, window=10, decay=0.85, super_recent=5)
    league_positions = calculate_league_position_stats(df)
    team_tiers = calculate_team_tiers(df)
    
    df["home_form"] = df.apply(lambda row: team_form[row["HomeTeam"]].get(row["Date"], 1.5), axis=1)
    df["away_form"] = df.apply(lambda row: team_form[row["AwayTeam"]].get(row["Date"], 1.5), axis=1)
    
    df["home_goals_scored_avg"] = df.apply(lambda row: goal_stats[row["HomeTeam"]][row["Date"]]["scored"], axis=1)
    df["home_goals_conceded_avg"] = df.apply(lambda row: goal_stats[row["HomeTeam"]][row["Date"]]["conceded"], axis=1)
    df["away_goals_scored_avg"] = df.apply(lambda row: goal_stats[row["AwayTeam"]][row["Date"]]["scored"], axis=1)
    df["away_goals_conceded_avg"] = df.apply(lambda row: goal_stats[row["AwayTeam"]][row["Date"]]["conceded"], axis=1)
    
    df["h2h_home_win_rate"] = df.apply(lambda row: h2h[(row["HomeTeam"], row["AwayTeam"])]["home_wins"] / max(h2h[(row["HomeTeam"], row["AwayTeam"])]["total"], 1), axis=1)
    df["h2h_away_win_rate"] = df.apply(lambda row: h2h[(row["HomeTeam"], row["AwayTeam"])]["away_wins"] / max(h2h[(row["HomeTeam"], row["AwayTeam"])]["total"], 1), axis=1)
    df["h2h_draw_rate"] = df.apply(lambda row: h2h[(row["HomeTeam"], row["AwayTeam"])]["draws"] / max(h2h[(row["HomeTeam"], row["AwayTeam"])]["total"], 1), axis=1)
    
    df["home_home_form"] = df.apply(lambda row: home_away_stats[row["HomeTeam"]][row["Date"]]["home_form"], axis=1)
    df["away_away_form"] = df.apply(lambda row: home_away_stats[row["AwayTeam"]][row["Date"]]["away_form"], axis=1)
    
    df["home_elo"] = df.apply(lambda row: elo_ratings[row["HomeTeam"]].get(row["Date"], 1500.0), axis=1)
    df["away_elo"] = df.apply(lambda row: elo_ratings[row["AwayTeam"]].get(row["Date"], 1500.0), axis=1)
    
    df["home_win_streak"] = df.apply(lambda row: streaks[row["HomeTeam"]][row["Date"]]["win_streak"], axis=1)
    df["away_win_streak"] = df.apply(lambda row: streaks[row["AwayTeam"]][row["Date"]]["win_streak"], axis=1)
    df["home_unbeaten_streak"] = df.apply(lambda row: streaks[row["HomeTeam"]][row["Date"]]["unbeaten_streak"], axis=1)
    df["away_unbeaten_streak"] = df.apply(lambda row: streaks[row["AwayTeam"]][row["Date"]]["unbeaten_streak"], axis=1)
    df["home_loss_streak"] = df.apply(lambda row: streaks[row["HomeTeam"]][row["Date"]]["loss_streak"], axis=1)
    df["away_loss_streak"] = df.apply(lambda row: streaks[row["AwayTeam"]][row["Date"]]["loss_streak"], axis=1)
    
    df["home_weighted_form"] = df.apply(lambda row: weighted_form[row["HomeTeam"]].get(row["Date"], 1.5), axis=1)
    df["away_weighted_form"] = df.apply(lambda row: weighted_form[row["AwayTeam"]].get(row["Date"], 1.5), axis=1)
    
    df["home_league_position"] = df.apply(lambda row: league_positions[row["HomeTeam"]][row["Date"]]["position"], axis=1)
    df["away_league_position"] = df.apply(lambda row: league_positions[row["AwayTeam"]][row["Date"]]["position"], axis=1)
    df["home_points"] = df.apply(lambda row: league_positions[row["HomeTeam"]][row["Date"]]["points"], axis=1)
    df["away_points"] = df.apply(lambda row: league_positions[row["AwayTeam"]][row["Date"]]["points"], axis=1)
    
    df["home_tier"] = df["HomeTeam"].map(team_tiers).fillna(2)
    df["away_tier"] = df["AwayTeam"].map(team_tiers).fillna(2)
    df["tier_diff"] = df["away_tier"] - df["home_tier"]
    df["tier_matchup"] = df["home_tier"] * 10 + df["away_tier"]
    
    df["form_diff"] = df["home_form"] - df["away_form"]
    df["goal_diff"] = (df["home_goals_scored_avg"] - df["home_goals_conceded_avg"]) - (df["away_goals_scored_avg"] - df["away_goals_conceded_avg"])
    df["home_advantage"] = df["home_home_form"] - df["away_away_form"]
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["position_diff"] = df["away_league_position"] - df["home_league_position"]
    df["points_diff"] = df["home_points"] - df["away_points"]
    df["weighted_form_diff"] = df["home_weighted_form"] - df["away_weighted_form"]
    df["streak_momentum"] = (df["home_win_streak"] - df["away_win_streak"]) + (df["away_loss_streak"] - df["home_loss_streak"])
    
    df["goal_expectation_home"] = df["home_goals_scored_avg"] * (1 - df["away_goals_conceded_avg"] / 3.0)
    df["goal_expectation_away"] = df["away_goals_scored_avg"] * (1 - df["home_goals_conceded_avg"] / 3.0)
    df["goal_expectation_diff"] = df["goal_expectation_home"] - df["goal_expectation_away"]
    
    df["home_power"] = (df["home_elo"] / 100) + df["home_weighted_form"] * 2.0 + (20 - df["home_league_position"]) - (df["home_tier"] * 3)
    df["away_power"] = (df["away_elo"] / 100) + df["away_weighted_form"] * 2.0 + (20 - df["away_league_position"]) - (df["away_tier"] * 3)
    df["power_diff"] = df["home_power"] - df["away_power"]
    
    df["elo_form_interaction"] = df["elo_diff"] * df["weighted_form_diff"]
    df["position_form_interaction"] = df["position_diff"] * df["form_diff"]
    df["tier_form_interaction"] = df["tier_diff"] * df["weighted_form_diff"]
    
    print(f"Features engineered. Total features: {len(df.columns)}")
    return df


def build_ensemble_model(all_teams: List[str], calibrate: bool = True) -> Pipeline:
    categorical_features = ["HomeTeam", "AwayTeam"]
    numeric_features = [
        "home_form", "away_form",
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
        "home_home_form", "away_away_form",
        "home_elo", "away_elo",
        "home_win_streak", "away_win_streak",
        "home_unbeaten_streak", "away_unbeaten_streak",
        "home_loss_streak", "away_loss_streak",
        "home_weighted_form", "away_weighted_form",
        "home_league_position", "away_league_position",
        "home_points", "away_points",
        "form_diff", "goal_diff", "home_advantage",
        "elo_diff", "position_diff", "points_diff",
        "weighted_form_diff", "streak_momentum",
        "goal_expectation_home", "goal_expectation_away", "goal_expectation_diff",
        "home_power", "away_power", "power_diff",
        "elo_form_interaction", "position_form_interaction"
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("teams", OneHotEncoder(handle_unknown="ignore", categories=[all_teams, all_teams]), categorical_features),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), numeric_features),
        ],
        remainder="drop",
    )
    
    class_weights = 'balanced'
    
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='log2',
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=350,
        learning_rate=0.055,
        max_depth=10,
        min_samples_split=2,
        subsample=0.9,
        max_features='log2',
        random_state=42
    )
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(768, 384, 192, 96),
        activation="relu",
        solver="adam",
        alpha=5e-7,
        batch_size=32,
        learning_rate="adaptive",
        learning_rate_init=3e-4,
        max_iter=500,
        random_state=42,
        early_stopping=False,
        momentum=0.95,
        nesterovs_momentum=True
    )
    
    estimators = [
        ("rf", rf),
        ("gb", gb),
        ("mlp", mlp)
    ]
    
    if XGBOOST_AVAILABLE:
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.035,
            max_depth=12,
            min_child_weight=1,
            subsample=0.93,
            colsample_bytree=0.93,
            colsample_bylevel=0.92,
            gamma=0.01,
            reg_alpha=0.01,
            reg_lambda=2.0,
            scale_pos_weight=1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        estimators.append(("xgb", xgb))
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=[3, 6, 1, 8] if XGBOOST_AVAILABLE else [3, 6, 1],
        n_jobs=-1
    )
    
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", ensemble),
    ])
    
    return pipe


def train_and_evaluate(df: pd.DataFrame, use_ensemble: bool = True) -> Tuple[Pipeline, dict]:
    df = engineer_features(df)
    
    feature_cols = [
        "HomeTeam", "AwayTeam",
        "home_form", "away_form",
        "home_goals_scored_avg", "home_goals_conceded_avg",
        "away_goals_scored_avg", "away_goals_conceded_avg",
        "h2h_home_win_rate", "h2h_away_win_rate", "h2h_draw_rate",
        "home_home_form", "away_away_form",
        "home_elo", "away_elo",
        "home_win_streak", "away_win_streak",
        "home_unbeaten_streak", "away_unbeaten_streak",
        "home_loss_streak", "away_loss_streak",
        "home_weighted_form", "away_weighted_form",
        "home_league_position", "away_league_position",
        "home_points", "away_points",
        "home_tier", "away_tier",
        "tier_diff", "tier_matchup",
        "form_diff", "goal_diff", "home_advantage",
        "elo_diff", "position_diff", "points_diff",
        "weighted_form_diff", "streak_momentum",
        "goal_expectation_home", "goal_expectation_away", "goal_expectation_diff",
        "home_power", "away_power", "power_diff",
        "elo_form_interaction", "position_form_interaction", "tier_form_interaction"
    ]
    
    X = df[feature_cols].copy()
    y = df["FTR"].astype(str)
    
    all_teams = pd.unique(pd.concat([X["HomeTeam"], X["AwayTeam"]], ignore_index=True)).tolist()
    all_teams = sorted(all_teams)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    print("\nTraining ensemble model...")
    pipe = build_ensemble_model(all_teams, calibrate=False)
    
    X_train_base, X_calib, y_train_base, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Base training: {len(X_train_base)} samples, Calibration: {len(X_calib)} samples")
    print("Training base ensemble...")
    pipe.fit(X_train_base, y_train_base)
    
    print("Calibrating probabilities...")
    pipe_calibrated = CalibratedClassifierCV(pipe, method='isotonic', cv='prefit')
    pipe_calibrated.fit(X_calib, y_calib)
    
    y_proba_calib = pipe_calibrated.predict_proba(X_test)
    
    y_pred = []
    classes = pipe.classes_
    
    for i in range(len(X_test)):
        proba = y_proba_calib[i]
        sorted_idx = np.argsort(proba)[::-1]
        top_prob = proba[sorted_idx[0]]
        second_prob = proba[sorted_idx[1]]
        
        if top_prob < 0.50 and (top_prob - second_prob) < 0.15:
            draw_idx = list(classes).index('D') if 'D' in classes else -1
            if draw_idx != -1 and proba[draw_idx] > 0.20:
                y_pred.append('D')
            else:
                y_pred.append(classes[sorted_idx[0]])
        else:
            y_pred.append(classes[sorted_idx[0]])
    
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classes_order": pipe.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=pipe.classes_).tolist(),
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
        "teams": all_teams,
        "feature_columns": feature_cols,
        "calibrated": True,
        "draw_aware": True,
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return pipe_calibrated, metrics


def load_dataset(path: str) -> pd.DataFrame:
    """Load and basic clean the dataset"""
    df = pd.read_csv(path)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["FTHG", "FTAG"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def save_model(pipe: Pipeline, metrics: dict, path: str) -> None:
    """Save trained model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "pipeline": pipe,
        "metrics": metrics,
    }
    joblib.dump(payload, path)
    print(f"\n✓ Model saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train model for a league")
    parser.add_argument("--league", type=str, required=True, help="League name (e.g., EFL-Championship)")
    parser.add_argument("--data-file", type=str, required=True, help="Data CSV file name")
    parser.add_argument("--model-name", type=str, default=None, help="Model file name (default: best_model_{league}.pkl)")
    args = parser.parse_args()
    
    # Setup paths - handle both relative to script location and current working directory
    script_dir = os.path.dirname(__file__)
    # Try relative to current working directory first (for extended script)
    league_dir = os.path.join(os.getcwd(), args.league)
    if not os.path.exists(league_dir):
        # Fall back to script directory (for original usage)
        league_dir = os.path.join(script_dir, args.league)
    
    data_path = os.path.join(league_dir, args.data_file)
    models_dir = os.path.join(league_dir, "models")
    
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"best_model_{args.league.lower().replace('-', '_')}.pkl"
    
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Training model for {args.league}")
    print(f"Data file: {data_path}")
    print(f"Model will be saved to: {model_path}")
    
    # Load data
    print("\nLoading dataset...")
    df = load_dataset(data_path)
    print(f"Loaded {len(df)} matches from {df['Date'].min()} to {df['Date'].max()}")
    
    # Train model
    pipe, metrics = train_and_evaluate(df, use_ensemble=True)
    
    # Save model
    save_model(pipe, metrics, model_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {model_path}")
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

