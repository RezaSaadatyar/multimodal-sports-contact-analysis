# Data Setup

## Dataset

The first version of this project is designed for the Kaggle competition:

**1st and Future - Player Contact Detection**

The dataset contains player tracking, contact labels, video, and video-related metadata.

## Important

The raw competition data are not stored in this GitHub repository.

Before downloading the data:

1. Sign in to Kaggle.
2. Open the competition page.
3. Accept the competition rules.
4. Install and configure the Kaggle command-line tool.

## Download

From the repository root:

```bash
mkdir -p data/raw
kaggle competitions download -c nfl-player-contact-detection -p data/raw
```

Unzip the downloaded archive into `data/raw/`.

A typical local structure will contain files such as:

```text
data/raw/
├── train_baseline_helmets.csv
├── train_labels.csv
├── train_player_tracking.csv
├── train_video_metadata.csv
├── train/
└── test/
```

The exact files should be checked after download because Kaggle competition assets can differ between train and test folders.

## Repository Rule

Do not commit raw videos, competition CSV files, or generated model checkpoints to GitHub.

The `.gitignore` file excludes these folders by default.
