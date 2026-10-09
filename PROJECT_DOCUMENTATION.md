# Project Documentation: Rental Price Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

Companion to `SDLC_DOCUMENTATION.md` and `README.md`.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Suggest a fair starting nightly price for a short-term rental listing |
| **Client context** | PropTech / short-term rental marketplace |
| **Data source** | NYC Airbnb Open Data — [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) |
| **Dataset size** | 48,895 listings → 48,870 after cleaning |
| **Final model** | Random Forest |
| **Test R²** | 0.160 (honestly low, expected in advance) |
| **Business framing** | % of predictions within a wider, realistic tolerance band |

---

## 2. Architecture

```
data/airbnb_nyc.csv (48,895 real 2019 NYC listings)
        │
        ▼
data_loader.py ──────► loads raw data, flags 3 real quality issues
        │
        ▼
clean_and_engineer.py ─► drops 11 $0-price listings, drops 14
        │                1,000+-night-minimum listings, imputes
        │                reviews_per_month=0 (informative missingness,
        │                NOT an error), engineers has_reviews
        ▼
   split.py ───────────► plain random split (no time axis, regression)
        │
        ▼
train_model.py ───────► Linear Regression (baseline) → Random Forest →
        │                XGBoost, trained on log(price), evaluated with
        │                a WIDENED tolerance band (20%, not Day 17's 10%)
        ▼
outputs/rental_price_model.joblib
        │
        ├──────────────► app.py ─── FastAPI service returning a PRICE
        │                           RANGE, not a point estimate
        │
        └──────────────► monitor.py ─ market-drift + regulatory-shock
                                       monitoring
```

---

## 3. Data Dictionary

| Column | Type | Description | Treatment |
|---|---|---|---|
| `neighbourhood_group` | string | NYC borough | One-hot encoded |
| `room_type` | string | Entire home/apt, Private room, Shared room | One-hot encoded |
| `latitude`, `longitude` | float | Listing coordinates | Kept as-is — the geospatial signal |
| `price` | int | Nightly price (USD) | **Target**, log-transformed for training |
| `minimum_nights` | int | Minimum required stay | 14 implausible (>365) values excluded |
| `number_of_reviews` | int | Total review count | Kept as-is |
| `reviews_per_month` | float | Review rate | 10,052 missing values imputed as 0 (verified informative, not an error) |
| `calculated_host_listings_count` | int | Host's total listing count | Kept as-is |
| `availability_365` | int | Days available per year | Kept as-is |

**Engineered feature:**

| Feature | Formula | Rationale |
|---|---|---|
| `has_reviews` | 1 if number_of_reviews > 0 | A brand-new listing is qualitatively different from an established low-review-rate one |

**Dropped columns:** `id`, `host_id` (unique identifiers), `name`,
`host_name`, `last_review` (free text/dates, not used as features — a
stated scope simplification).

---

## 4. Key EDA Findings

| Finding | Detail |
|---|---|
| Price by borough | Manhattan ($150 median) > Brooklyn ($90) > Queens/Staten Island ($75) > Bronx ($65) |
| Price by room type | Entire home/apt ($160) > Private room ($70) > Shared room ($45) |
| Manhattan entire apt vs. Staten Island shared room | $191 vs. $30 median — a 6.4x spread |
| **Counter-intuitive: reviews and price** | Listings WITHOUT reviews have a HIGHER median price ($120) than reviewed ones ($101) |
| availability_365 correlation with price | 0.082 — essentially no linear relationship |

---

## 5. Modeling Results

### Experiment log (validation set, real USD)

| Model | RMSE | MAE | R² | Within 20% |
|---|---|---|---|---|
| Linear Regression (baseline) | $247.90 | $61.75 | 0.078 | 37.8% |
| **Random Forest** | **$241.28** | **$56.18** | **0.127** | **43.5%** |
| XGBoost | $242.06 | $56.40 | 0.121 | 43.2% |

### Test set (final, honest read)

| Metric | Value |
|---|---|
| Test RMSE | $224.57 |
| Test MAE | $57.83 |
| **Test R²** | **0.160** |
| Within ±10% | 22.7% |
| Within ±20% | 43.2% |

**Why this R² is honestly reported as expected, not disappointing:**
stated explicitly in `train_model.py` BEFORE training — a rental's price
depends heavily on photos, host responsiveness, and real-time seasonal
demand, none of which exist in this dataset. A low R² here reflects a
genuinely harder, more information-poor problem than Day 17's car
pricing (R²=0.94), not a worse model or weaker effort.

### What drives the model

Top features: `room_type_Entire home/apt` (0.502, by far the dominant
signal), `longitude` (0.141), `latitude` (0.118), `availability_365`
(0.066) — the geospatial coordinates alone account for over a quarter
of total feature importance, confirming EDA's borough-level price
pattern is a real, learnable signal.

---

## 6. API Reference

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser test form |
| `/docs` | GET | Interactive Swagger UI |
| `/health` | GET | Health check |
| `/price` | POST | Get a price suggestion for a listing |

**Response — a RANGE, not a point estimate:**
```json
{"suggested_price_usd": 268.75, "typical_range_usd": [211.02, 326.48], "top_factors": ["room_type_Entire home/apt", "longitude", "latitude"]}
```

**Why a range, not one number:** given the model's honestly-measured
0.16 R², presenting a single confident-looking price would overstate
its actual precision. The range is built directly from the model's own
test-set MAE, not an invented confidence interval.

Verified with two constructed listings: a Manhattan entire apartment
suggested $268.75 ($211–$326 range); a Bronx shared room suggested
$39.95 ($0–$98 range, the wide band reflecting genuine uncertainty at
low price points).

---

## 7. Monitoring & Maintenance Plan

`monitor.py` checks PSI drift on `availability_365`, `reviews_per_month`,
and `minimum_nights`, plus MAE decay.

**Recommended cadence: monthly-to-quarterly for gradual demand drift**
(same general category as Day 17's used-car market) **PLUS a
regulatory-shock trigger** Day 17 doesn't need — NYC's short-term-rental
rules changed materially after this 2019 snapshot, and a policy change
can shift this market discontinuously, not gradually, which a fixed
monitoring calendar alone wouldn't catch in time.

---

## 8. Known Limitations

1. **R²=0.16 reflects real, unavoidable information gaps** in the dataset (no photos, host quality, or seasonal demand) — not a modeling shortfall, and stated as expected before training.
2. **2019 NYC snapshot only** — not validated against a different city, time period, or regulatory environment.
3. **Only 22.7% of predictions land within ±10%**, though 43.2% land within the wider, more realistic ±20% band used as this project's primary tolerance.
4. **No text-mining of the `name` field**, which could plausibly carry additional price signal (e.g., "luxury," "cozy," "near subway") — a stated scope simplification.

---

## 9. File Map

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, flag 3 quality issues |
| `eda.py` | 6 | Borough/room-type patterns, geospatial check |
| `clean_and_engineer.py` | 5 (fixes) + 7 | The 3 fixes, feature engineering |
| `split.py` | 7 | Random split |
| `train_model.py` | 8–9 | Model comparison, honestly-low-R² validation |
| `app.py` | 10 | FastAPI service, price-range response |
| `monitor.py` | 13 | Market-drift + regulatory-shock monitoring |
| `README.md` | 12 | Setup instructions, Day 17 comparison |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation |
| `SDLC_DOCUMENTATION.md` | 1–14 | Full 14-phase narrative |
