"""Shared preprocessing and scikit-learn pipeline definitions for the ATP Tennis dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

SERIES_MAP = {
    "International": "ATP250",
    "International Gold": "ATP500",
    "Masters": "Masters 1000",
}
SERIES_ORDER = ["ATP250", "ATP500", "Masters 1000", "Masters Cup", "Grand Slam"]
ROUND_ORDER = [
    "1st Round",
    "2nd Round",
    "3rd Round",
    "4th Round",
    "Quarterfinals",
    "Semifinals",
    "The Final",
]
DROP_COLUMNS = [
    "Tournament",
    "Date",
    "Pts_1",
    "Pts_2",
    "Rank_1",
    "Rank_2",
    "Odd_1",
    "Odd_2",
]

NUM_FEATURES = ["log_rank_ratio", "log_rank_mean", "implied_prob_1"]
ORD_FEATURES = ["Series", "Round", "Best of"]
ORD_CATEGORIES = [SERIES_ORDER, ROUND_ORDER, [3, 5]]
NOM_FEATURES = ["Surface", "Court"]
FINAL_FEATURES = NUM_FEATURES + ORD_FEATURES + NOM_FEATURES


def merge_series(df: pd.DataFrame) -> pd.DataFrame:
    """Merge pre-2009 ATP tournament category names into their modern equivalents."""
    df = df.copy()
    df["Series"] = df["Series"].replace(SERIES_MAP)
    return df


def map_round_robin(df: pd.DataFrame) -> pd.DataFrame:
    """Map Round Robin matches into the ordinal Round hierarchy."""
    df = df.copy()
    is_rr = df["Round"] == "Round Robin"
    df.loc[is_rr & (df["Series"] == "Masters Cup"), "Round"] = "Quarterfinals"
    df.loc[is_rr & (df["Series"] != "Masters Cup"), "Round"] = "1st Round"
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comparative player features (log_rank_ratio, log_rank_mean, implied_prob_1)."""
    df = df.copy()
    log_rank_1, log_rank_2 = np.log(df["Rank_1"]), np.log(df["Rank_2"])
    df["log_rank_ratio"] = log_rank_2 - log_rank_1
    df["log_rank_mean"] = (log_rank_1 + log_rank_2) / 2
    p1, p2 = 1 / df["Odd_1"], 1 / df["Odd_2"]
    df["implied_prob_1"] = p1 / (p1 + p2)
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all Part 2 deterministic cleaning and feature engineering steps."""
    df = merge_series(df)
    df = map_round_robin(df)
    df = add_features(df)
    return df.drop(columns=DROP_COLUMNS)


def build_pipeline() -> Pipeline:
    """Build the full Part 3 scikit-learn preprocessing Pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    ord_encoder = OrdinalEncoder(categories=ORD_CATEGORIES)
    nom_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUM_FEATURES),
            ("ord", ord_encoder, ORD_FEATURES),
            ("nom", nom_encoder, NOM_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([
        ("prepare_features", FunctionTransformer(prepare_features)),
        ("preprocessor", preprocessor),
    ])

