"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY THE RIGHT CADENCE HERE IS SIMILAR TO DAY 17's, BUT FOR A PARTLY
DIFFERENT REASON: like the used-car market, the short-term rental
market drifts gradually with the seasons and local events -- but it
ALSO drifts with something Day 17's car market doesn't face nearly as
much: local short-term-rental REGULATION (NYC's own rules on Airbnb
listings changed significantly after this 2019 snapshot). A monitoring
system for this kind of model needs to watch for regulatory shifts, not
just gradual demand drift, since a regulation change can shift the
market discontinuously rather than gradually.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data

MAE_INCREASE_THRESHOLD_PCT = 0.20


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/rental_price_model.joblib")

    raw = load_raw_data()
    engineered = engineer_features(clean_data(raw))
    cat_cols, num_cols = get_feature_columns()
    feature_cols = cat_cols + num_cols
    X = engineered[feature_cols]
    y_log = np.log1p(engineered["price"])
    y_raw = engineered["price"]

    X_train, X_val, X_test, y_log_train, y_log_val, y_log_test = split_data(X, y_log)
    y_raw_test = y_raw.loc[y_log_test.index]

    print("=== Feature Drift (PSI): availability, reviews, minimum_nights ===")
    for feat in ["availability_365", "reviews_per_month", "minimum_nights"]:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")

    test_pred = np.expm1(model.predict(X_test))
    current_mae = mean_absolute_error(y_raw_test, test_pred)
    print(f"\n=== Performance on test batch ===")
    print(f"MAE: ${current_mae:.2f}")

    with open("outputs/business_validation.json") as f:
        baseline = json.load(f)
    baseline_mae = baseline["test_mae_usd"]
    pct_change = (current_mae - baseline_mae) / baseline_mae

    if pct_change > MAE_INCREASE_THRESHOLD_PCT:
        print(f"\n⚠️  RETRAIN TRIGGERED: MAE rose {pct_change:+.1%} (threshold: {MAE_INCREASE_THRESHOLD_PCT:.0%})")
    else:
        print(f"\n✅ No retrain needed (MAE change: {pct_change:+.1%}, threshold: {MAE_INCREASE_THRESHOLD_PCT:.0%})")

    print("\n=== Recommended monitoring cadence ===")
    print("Monthly-to-quarterly for gradual demand drift, SAME general cadence")
    print("category as Day 17's used-car market -- but ALSO needs an event-driven")
    print("trigger this project's other pricing model doesn't: a local regulation")
    print("change (NYC's short-term-rental rules shifted materially after this 2019")
    print("snapshot) can move this market in a single policy change, not gradually,")
    print("which a fixed monitoring calendar alone wouldn't catch in time.")


if __name__ == "__main__":
    run_monitoring_check()
