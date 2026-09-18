# Multimodal Sports Contact Analysis

## 🚀 Project Overview
This repository develops a step-by-step pipeline for analysing player contact events using **player tracking**, **video-derived information**, and **multimodal machine learning**.

The project starts with interpretable tracking features, then adds video information, and finally compares tracking-only, video-only, and multimodal models. The main research question is:

> **Does video provide complementary information beyond player tracking alone for contact-event detection?**

**Maintained by:** Reza Saadatyar (2026)  
**Core Technologies:** Python, PyTorch, Pandas, NumPy, Scikit-Learn, OpenCV

---

## 🎯 Project Objectives
- Load and audit player-tracking and contact-label data.
- Build player-pair features such as distance, relative speed, acceleration, and orientation.
- Develop a PyTorch tracking-only baseline.
- Add video-derived features in a separate stage.
- Develop a multimodal PyTorch model combining tracking and video information.
- Evaluate rare contact events with class-aware metrics and grouped validation.
- Compare tracking-only, video-only, and multimodal performance under the same data split.

---

## 📊 Data Source
The initial implementation is designed around the **1st and Future - Player Contact Detection** dataset hosted on Kaggle.

The competition data include:
- player tracking at 10 Hz;
- contact labels for player-player and player-ground events;
- sideline and endzone video;
- video metadata and helmet detections.

The raw competition files are **not included in this repository**. Access is subject to the Kaggle competition rules.

See [data/README.md](data/README.md) for the download and folder setup.

---

## 🧪 Planned Experimental Stages

| Stage | Focus | Main Output |
| :--- | :--- | :--- |
| **Stage 1** | Data audit and exploration | Clean understanding of tracking, labels, and missingness |
| **Stage 2** | Tracking feature engineering | Event-level player-pair feature table |
| **Stage 3** | Tracking-only modelling | PyTorch baseline and class-aware evaluation |
| **Stage 4** | Video processing | Video-derived spatial and temporal features |
| **Stage 5** | Multimodal modelling | Tracking + video fusion model |
| **Stage 6** | Comparative evaluation | Tracking-only vs video-only vs multimodal results |

---

## 📂 Repository Map

### 🐍 [code/python/](code/python/) (Primary Python & PyTorch Pipeline)
- **[src/core/](code/python/src/core/)**: training, evaluation, and experiment logic.
- **[src/models/](code/python/src/models/)**: PyTorch model definitions.
- **[src/utils/](code/python/src/utils/)**: data loading, reproducibility, hardware checks, and feature utilities.
- **[notebooks/](code/python/notebooks/)**: sequential notebooks from data exploration to multimodal evaluation.

### 📁 [data/](data/)
- `raw/`: locally downloaded Kaggle files. Not committed to Git.
- `processed/`: generated intermediate datasets. Not committed to Git.

### 📄 [docs/](docs/)
- technical notes, data dictionary, modelling decisions, and experiment documentation.

### 📈 [outputs/](outputs/)
- generated figures, metrics, reports, and model comparisons.

### 💾 [checkpoints/](checkpoints/)
- local model checkpoints. Not committed to Git.

---

## 🛠️ Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/RezaSaadatyar/multimodal-sports-contact-analysis.git
   cd multimodal-sports-contact-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Install the requirements**
   ```bash
   pip install -r code/python/requirements.txt
   ```

4. **Download the competition data**
   Follow the instructions in [data/README.md](data/README.md).

5. **Run the notebooks in order**
   Start with:
   ```text
   code/python/notebooks/01_data_exploration.ipynb
   ```

---

## ⚖️ Evaluation Strategy
The project will not rely on accuracy alone because contact events are imbalanced.

Primary evaluation will include:
- sensitivity / recall;
- specificity;
- precision / PPV;
- F1 score;
- AUROC;
- AUPRC;
- Matthews correlation coefficient;
- false-positive and false-negative analysis.

Data splitting will be grouped at the `game_play` level to reduce leakage between training and evaluation data.

---

## 🔬 Research Scope
This is an **independent research and portfolio project** using publicly accessible competition data. It is not affiliated with Hawk-Eye Innovations or Leeds Beckett University and does not use proprietary Hawk-Eye data.

The repository is intended to document the full modelling process from data understanding to reproducible multimodal evaluation.