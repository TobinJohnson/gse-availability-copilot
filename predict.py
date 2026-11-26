"""
7-day failure-risk model: which units are most likely to break next week?

Features per unit and week: age, run hours in the last 28 days, days since
last preventive maintenance, failures in the last 90 days, season, and
equipment type. Label: did the unit fail in the next 7 days?

Model: logistic regression (simple, explainable to a maintenance manager).
It is trained on the older weeks and tested on the most recent 13 weeks it
never saw, so the accuracy numbers are honest.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age_years", "run_hours_28d", "days_since_pm", "failures_90d", "summer", "monsoon"]
CATEGORICAL = ["type_code"]
TEST_WEEKS = 13


def _features(df):
    df = df.copy()
    df["summer"] = df["month"].isin([4, 5, 6]).astype(int)
    df["monsoon"] = df["month"].isin([7, 8, 9]).astype(int)
    return df


def train(unit_weeks: pd.DataFrame):
    df = _features(unit_weeks.dropna(subset=["failed_next_7d"]))
    weeks = sorted(df["week_start"].unique())
    cutoff = weeks[-TEST_WEEKS]
    train_df, test_df = df[df["week_start"] < cutoff], df[df["week_start"] >= cutoff]

    model = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), NUMERIC),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train_df[NUMERIC + CATEGORICAL], train_df["failed_next_7d"].astype(int))

    p = model.predict_proba(test_df[NUMERIC + CATEGORICAL])[:, 1]
    y = test_df["failed_next_7d"].astype(int).to_numpy()
    top = p >= pd.Series(p).quantile(0.90)           # the 10% riskiest unit-weeks
    metrics = {
        "auc": roc_auc_score(y, p),
        "base_rate": y.mean(),
        "top10_precision": y[top].mean(),
        "top10_capture": y[top].sum() / max(1, y.sum()),
        "test_period": f"{cutoff} onwards ({TEST_WEEKS} weeks)",
    }
    return model, metrics


def current_risk(model, unit_weeks: pd.DataFrame, equipment: pd.DataFrame) -> pd.DataFrame:
    """Risk for the week after the snapshot (the latest week, which has no label yet)."""
    latest = unit_weeks[unit_weeks["week_start"] == unit_weeks["week_start"].max()]
    df = _features(latest)
    df["risk_7d"] = model.predict_proba(df[NUMERIC + CATEGORICAL])[:, 1]
    df = df.merge(equipment[["unit_id", "type_name"]], on="unit_id")
    return df.sort_values("risk_7d", ascending=False)


def drivers(model) -> pd.DataFrame:
    """Logistic-regression coefficients for the numeric features (on the standardised scale)."""
    coefs = model.named_steps["clf"].coef_[0][:len(NUMERIC)]
    return pd.DataFrame({"feature": NUMERIC, "weight": coefs}).sort_values("weight", ascending=False)
