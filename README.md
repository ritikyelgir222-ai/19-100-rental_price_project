# Rental Price Prediction — Day 19 (PropTech, ML)

Full project code for Day 19 of the 100-day series. Uses the real
**New York City Airbnb Open Data** (Kaggle: dgomonov/new-york-city-airbnb-open-data,
sourced from Inside Airbnb): 48,895 real 2019 NYC short-term rental listings.

## How this compares to Day 17 (Used Car Price Prediction)

Both are price-regression projects, but the underlying problem
difficulty is genuinely different — stated before training, not
discovered as an excuse afterward:

| Aspect | Day 17 (Used Cars) | Day 19 (Rentals) |
|---|---|---|
| Price driver visibility | Mechanical specs are fully observable (age, mileage, engine) | Photos, host quality, real-time seasonal demand — all real price drivers, **none observable in this data** |
| Expected R² | High (0.94 achieved) | **Low, expected in advance** (0.16 achieved) |
| Data quality issues | Whitespace, duplicates, a scraping gap | **Informative missingness** (blank reviews = "never reviewed," not lost data) + a platform-usage quirk (1,000+ night minimum stays) |
| Pricing tolerance used | ±10% ("fair pricing") | **±20%**, explicitly widened because a tighter bar would be unrealistic given what this data can see |
| API response | One price number | **A price range**, since a confident single number would overstate this model's honestly-measured precision |
| Monitoring trigger | Gradual market drift | Gradual drift **+ regulatory shocks** (NYC's short-term-rental rules changed materially after this 2019 snapshot) |

## Three real data quality findings

1. **11 listings priced $0** — genuinely invalid (named, described listings don't rent for free) — dropped
2. **14 listings requiring 1,000+ night minimum stays** — not malformed data, but a well-documented platform-usage pattern: hosts using an absurd minimum stay to effectively delist without removing the listing — excluded from the short-term-rental population this model targets
3. **10,052 listings missing `reviews_per_month`** — verified to *exactly* match `number_of_reviews == 0`: this is NOT a data quality problem, it's informative missingness ("never reviewed"), correctly imputed as 0 rather than treated as an error

## A genuinely counter-intuitive EDA finding

Listings **without** any reviews have a *higher* median price ($120)
than reviewed ones ($101) — plausibly new listings priced optimistically
before hosts adjust down based on booking demand. Reported directly,
not smoothed over.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py          # Phase 5 — loads real data, flags 3 quality issues
python eda.py                   # Phase 6 — borough/room-type price patterns, geospatial check
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — the 3 fixes, feature engineering
python train_model.py          # Phase 8-9 — linear vs. RF vs. XGBoost, honestly-low R²
python monitor.py              # Phase 13 — market-drift + regulatory-shock monitoring
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/** — response is a price *range*, not a
single number.

## File map

| File | SDLC Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Loads real data, flags 3 quality issues |
| `eda.py` | 6 | Borough/room-type patterns, review-price relationship |
| `clean_and_engineer.py` | 5 (fixes) + 7 | The 3 fixes (one of which isn't really a "fix"), feature engineering |
| `split.py` | 7 | Random split |
| `train_model.py` | 8–9 | Linear vs. RF vs. XGBoost, honestly-low-R² business validation |
| `app.py` | 10 | FastAPI service returning a price range |
| `monitor.py` | 13 | Market-drift + regulatory-shock monitoring cadence |

## Known limitations (stated honestly)

- **R² of 0.16 is expected, not a modeling failure** — no photos, host-quality signal, or real-time demand exist in this dataset, all genuine price drivers this model simply cannot see.
- **2019 snapshot, NYC only** — short-term-rental regulation has changed materially since; not assumed current or transferable to another city.
- **Only 22.7% of predictions land within ±10%**, though 43.2% land within ±20% — a much wider realistic tolerance than Day 17's cars.
