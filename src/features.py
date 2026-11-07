"""Part 2: data cleaning and feature engineering for the ATP tennis dataset.

All functions are stateless: they learn nothing from the data and do not use the target,
so prepare_features can be the first step of the scikit-learn pipeline (Part 3).
They live in this file (not in the notebook) so that the saved pipeline (joblib)
can be loaded in a new notebook in M2.

Usage from a notebook in notebooks/:
    import sys
    sys.path.append("..")   # project root
    from src.features import prepare_features
"""

import numpy as np
import pandas as pd

# section 10: old (until 2008) and new Series names
SERIES_MAP = {"International": "ATP250", "International Gold": "ATP500", "Masters": "Masters 1000"}
SERIES_ORDER = ["ATP250", "ATP500", "Masters 1000", "Masters Cup", "Grand Slam"]

# section 11: natural order of the rounds
ROUND_ORDER = ["1st Round", "2nd Round", "3rd Round", "4th Round", "Quarterfinals", "Semifinals", "The Final"]

# section 19: raw columns that are replaced by the new features or not used
DROP_COLUMNS = ["Tournament", "Date", "Pts_1", "Pts_2", "Rank_1", "Rank_2", "Odd_1", "Odd_2"]


def merge_series(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the pre-2009 Series names to the current ATP names."""
    df = df.copy()
    df["Series"] = df["Series"].replace(SERIES_MAP)
    return df


def map_round_robin(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 'Round Robin' with the round that has the same number of players left."""
    df = df.copy()
    round_robin = df["Round"] == "Round Robin"
    atp_finals = df["Series"] == "Masters Cup"
    df.loc[round_robin & atp_finals, "Round"] = "Quarterfinals"   # 8 players
    df.loc[round_robin & ~atp_finals, "Round"] = "1st Round"      # 2007 experiment, 24 players
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that compare the two players (NaN where an input is missing) and missing indicators."""
    df = df.copy()
    log_rank_1, log_rank_2 = np.log(df["Rank_1"]), np.log(df["Rank_2"])
    df["log_rank_ratio"] = log_rank_2 - log_rank_1           # = log(Rank_2 / Rank_1)
    df["log_rank_mean"] = (log_rank_1 + log_rank_2) / 2
    inv_1, inv_2 = 1 / df["Odd_1"], 1 / df["Odd_2"]
    df["implied_prob_1"] = inv_1 / (inv_1 + inv_2)
    # 0/1 indicators (section 15): one for the ranking features, one for the odds
    df["rank_missing"] = df["log_rank_ratio"].isna().astype(int)
    df["odds_missing"] = df["implied_prob_1"].isna().astype(int)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """All Part 2 steps; works on any dataframe with the Part 1 columns, does not use the target."""
    df = merge_series(df)
    df = map_round_robin(df)
    df = add_features(df)
    return df.drop(columns=DROP_COLUMNS)
