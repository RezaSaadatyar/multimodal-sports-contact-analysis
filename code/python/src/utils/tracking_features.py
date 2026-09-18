# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

import numpy as np
import pandas as pd


def add_player_pair_distance(df):
    """
    Calculate Euclidean distance between two players.

    Required columns:
        x_position_1, y_position_1, x_position_2, y_position_2
    """
    required = {
        "x_position_1",
        "y_position_1",
        "x_position_2",
        "y_position_2",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing position columns: {sorted(missing)}")

    out = df.copy()

    dx = out["x_position_1"] - out["x_position_2"]
    dy = out["y_position_1"] - out["y_position_2"]

    out["player_distance"] = np.sqrt(dx ** 2 + dy ** 2)
    return out


def add_relative_motion_features(df):
    """
    Add simple relative speed and acceleration features when both players have
    those variables available.

    Expected optional columns:
        speed_1, speed_2, acceleration_1, acceleration_2
    """
    out = df.copy()

    if {"speed_1", "speed_2"}.issubset(out.columns):
        out["relative_speed"] = (out["speed_1"] - out["speed_2"]).abs()

    if {"acceleration_1", "acceleration_2"}.issubset(out.columns):
        out["relative_acceleration"] = (
            out["acceleration_1"] - out["acceleration_2"]
        ).abs()

    return out


def select_numeric_features(df, feature_columns):
    """
    Return a clean numeric feature matrix for modelling.

    Missing values are retained here so that the notebook can inspect them
    explicitly before choosing an imputation strategy.
    """
    missing = [col for col in feature_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Requested feature column(s) not found: {missing}")

    return df[feature_columns].apply(pd.to_numeric, errors="coerce")
