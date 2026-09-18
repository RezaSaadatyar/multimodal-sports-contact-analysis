# ========================= Presented by: Reza Saadatyar (2026) ================================
# ============================= E-mail: R.Saadatyar90@gmail.com ==================================

from pathlib import Path

import pandas as pd


REQUIRED_TRACKING_COLUMNS = {
    "game_play",
    "nfl_player_id",
    "step",
    "x_position",
    "y_position",
}

REQUIRED_LABEL_COLUMNS = {
    "contact_id",
    "game_play",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
}


def _check_columns(df, required_columns, table_name):
    """
    Check that required columns exist before the analysis continues.
    """
    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(
            f"{table_name} is missing required column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_tracking_data(data_dir):
    """
    Load the player-tracking table.

    Args:
        data_dir (str/Path): Directory containing Kaggle competition files.

    Returns:
        pd.DataFrame: Player tracking data.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / "train_player_tracking.csv"

    print(f"◽ Loading tracking data from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Tracking file not found: {file_path}\n"
            "Download and extract the Kaggle competition data first."
        )

    tracking = pd.read_csv(file_path)
    _check_columns(tracking, REQUIRED_TRACKING_COLUMNS, "Tracking table")

    print(f"✅ Tracking rows: {len(tracking):,}")
    print(f"✅ Tracking columns: {tracking.shape[1]}")
    return tracking


def load_contact_labels(data_dir):
    """
    Load the contact-label table.

    Args:
        data_dir (str/Path): Directory containing Kaggle competition files.

    Returns:
        pd.DataFrame: Contact labels.
    """
    data_dir = Path(data_dir)
    file_path = data_dir / "train_labels.csv"

    print(f"◽ Loading contact labels from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {file_path}\n"
            "Download and extract the Kaggle competition data first."
        )

    labels = pd.read_csv(file_path)
    _check_columns(labels, REQUIRED_LABEL_COLUMNS, "Contact-label table")

    labels["contact"] = labels["contact"].astype(int)

    print(f"✅ Label rows: {len(labels):,}")
    print(f"✅ Positive contacts: {labels['contact'].sum():,}")
    return labels


def load_competition_tables(data_dir):
    """
    Load the two tables required for the first tracking-only analysis.
    """
    tracking = load_tracking_data(data_dir)
    labels = load_contact_labels(data_dir)

    return tracking, labels
