# ATP Tennis Match Outcome Dataset Preparation


## Team
- Petar Car
- Karlo Vlašić
- Jakub Cerny

## Dataset
- Name: ATP Tennis 2000 - 2026 Daily update
- Source: https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull
- License: CC0 (Public Domain)
- Version used: downloaded on 26 Oct 2026, last match on 18 Oct 2026
- Size: 68,963 matches, 17 columns
- File: data/raw/atp_tennis.csv (never modified)
- Target: Player_1 wins the match (Winner == Player_1)

## Repository structure
- data/raw/: original dataset
- notebooks/: Jupyter notebooks
- src/: shared Python code
- report/: written report; report/figures/: figures
- slides/: presentation

## How to run
1. python -m venv venv
2. Activate the environment (Windows: venv\Scripts\activate, Mac/Linux: source venv/bin/activate)
3. pip install -r requirements.txt
4. jupyter notebook, then open the notebooks in order (01, 02, 03)

All random operations use random_state=42.
