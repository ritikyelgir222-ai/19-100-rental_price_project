# SDLC Documentation: Rental Price Prediction
### PropTech Industry | ML | Day 19 of the 100-Day Series
### Filled against the master 14-phase SDLC template, using the real project we built

Every number below comes from the actual pipeline run on the real NYC
Airbnb Open Data dataset.

---

## Phase 1: Discovery & Stakeholder Requirement Gathering
**Owner:** Business Analyst / Data Scientist | **Output:** Meeting notes, stakeholder map

- **Stakeholders identified:** marketplace pricing lead (problem owner), hosts listing properties (end users), guests (indirect beneficiaries of consistent pricing)
- **Discovery findings:**
  - Current process (assumed for this exercise): hosts set their own nightly price with limited data-driven guidance, leading to over- or under-pricing relative to comparable listings
  - Decision this project informs: what starting nightly price should a host be shown when creating or repricing a listing?
  - Critical framing established immediately: unlike Day 17's used cars, a rental's price is shaped by factors (photos, host responsiveness, real-time demand) this dataset fundamentally cannot see — this is stated as an expected limitation from the start, not discovered as a disappointment later
- **Deliverable — Problem Statement:** "Suggest a data-driven starting nightly price for a short-term rental listing, with honest uncertainty given what this data can and can't observe about what actually drives rental prices."

---

## Phase 2: Business Requirement Document (BRD)
**Owner:** Business Analyst | **Output:** BRD

- **Business objective:** provide hosts a defensible starting price, reducing the share of listings priced far outside the realistic market range for comparable properties
- **Scope:** in-scope: 2019 NYC listings across all 5 boroughs and 3 room types; out-of-scope: any other city or time period
- **Success metrics / KPIs:** % of predictions within a REALISTIC tolerance band (±20%, wider than Day 17's ±10%, justified explicitly by what this data can see)
- **Assumptions & constraints:** single 2019 snapshot; no photos, host-quality, or real-time demand signal available — a real ceiling on achievable accuracy, acknowledged upfront
- **Sign-off:** N/A — self-directed portfolio project

---

## Phase 3: Functional & Technical Requirement Document (FRD/TRD)
**Owner:** Data Scientist / Tech Lead | **Output:** FRD/TRD

- **Functional requirements:** given a listing's location, room type, and history, return a price suggestion AND an honest range, not a falsely-precise single number
- **Non-functional requirements:** sub-second response time (confirmed in testing)
- **Data sources:** a single static CSV in this exercise; a real system would need a live feed of recent comparable bookings to stay current
- **Integration points:** none in this version — a production version would integrate with the marketplace's listing-creation flow

---

## Phase 4: Project Planning
**Owner:** Project/Delivery Manager | **Output:** Project plan, risk register

- **Build sequence used:** data loading (flagging quality issues) → EDA (borough/room-type patterns) → cleaning/feature engineering → split → model comparison → API → monitoring → documentation
- **Real risks encountered during the build:**
  - Risk: the temptation to treat missing `reviews_per_month` as a standard imputation problem — resolved by first VERIFYING the missingness pattern (it exactly matches `number_of_reviews == 0`) before deciding how to handle it, revealing it wasn't actually a data quality issue at all
  - Risk: setting expectations for R² too high, given the series' prior pricing project (Day 17) achieved 0.94 — resolved by reasoning through what THIS dataset can and can't observe BEFORE training, so a lower result wouldn't be mistaken for a modeling failure

---

## Phase 5: Data Collection & Data Understanding
**Owner:** Data Engineer / Data Scientist | **Output:** Data dictionary, data quality report

- **Data inventory:** 48,895 real 2019 NYC Airbnb listings, 16 raw columns
- **Data quality report — three distinct findings, one of which isn't actually a data quality problem:**
  1. 11 listings priced $0 — genuine errors, dropped
  2. 14 listings requiring 1,000+ night minimum stays — a real platform-usage pattern (effective delisting), not malformed data, excluded from the short-term-rental population
  3. 10,052 listings missing `reviews_per_month` — verified to exactly match zero reviews, meaning this is INFORMATIVE missingness, not an error, and correctly imputed as 0 rather than treated as a gap to fill in some other way
- **Access & governance:** public, already-anonymized dataset; no additional governance required

---

## Phase 6: Exploratory Data Analysis (EDA)
**Owner:** Data Scientist | **Output:** `eda.py`, `outputs/eda_summary.png`

**Actual findings:**

- Manhattan entire home/apt: $191 median vs. Staten Island shared room: $30 median — a 6.4x spread
- Borough ranking: Manhattan > Brooklyn > Queens ≈ Staten Island > Bronx
- Room type ranking: Entire home/apt > Private room > Shared room, consistently across every borough
- **Genuinely counter-intuitive finding:** listings WITHOUT any reviews have a HIGHER median price ($120) than reviewed ones ($101) — plausibly new listings priced optimistically before hosts adjust down based on actual booking demand, reported directly rather than smoothed over
- availability_365 correlation with price: 0.082, essentially no linear relationship

---

## Phase 7: Data Preprocessing & Feature Engineering
**Owner:** Data Scientist / ML Engineer | **Output:** `clean_and_engineer.py`, `split.py`

- **Three distinct data handling decisions, each with different reasoning:** dropping $0-price listings (genuine errors, no principled imputation target), excluding 1,000+-night-minimum listings (a platform-usage pattern, not an error, but still outside the target population), and imputing missing `reviews_per_month` as 0 (the factually correct value once the missingness pattern was verified, not a data quality fix at all)
- **Feature engineering:** `has_reviews`, separating "never reviewed" as its own binary signal from the continuous review-rate value
- **Split strategy:** plain random split — same reasoning as Day 17, no time axis, no class imbalance

---

## Phase 8: Model Development
**Owner:** Data Scientist / ML Engineer | **Output:** `train_model.py`, `outputs/experiment_log.csv`

**Actual experiment log (validation set, real USD):**

| Model | RMSE | MAE | R² | Within 20% |
|---|---|---|---|---|
| Linear Regression (baseline) | $247.90 | $61.75 | 0.078 | 37.8% |
| **Random Forest** | **$241.28** | **$56.18** | **0.127** | **43.5%** |
| XGBoost | $242.06 | $56.40 | 0.121 | 43.2% |

**Why the tolerance band was widened to ±20% (vs. Day 17's ±10%), decided BEFORE seeing results:** stated explicitly in the code that a rental's price depends on factors this dataset cannot see, so a tighter bar would penalize the model for a limitation of the DATA, not the modeling approach — a methodological decision made in advance, not adjusted after seeing a disappointing number.

---

## Phase 9: Model Evaluation & Business Validation
**Owner:** Data Scientist + Business Stakeholder | **Output:** `outputs/business_validation.json`

- **Technical metrics (test set):** RMSE $224.57, MAE $57.83, **R² 0.160**
- **Business validation:** 22.7% of predictions within ±10%, 43.2% within ±20%
- **THE central honest finding of this project:** R²=0.16 is genuinely low compared to Day 17's R²=0.94 — and this is explained, not apologized for: a used car's price is driven almost entirely by observable mechanical specs, while a rental's price depends heavily on photos, host quality, and real-time demand, none of which exist in this dataset. A lower R² here reflects a harder, more information-poor problem, stated as an expected outcome from Phase 1 onward, not a shortfall discovered and excused after training.
- **Bias/fairness check:** not applicable in the demographic sense (no such features exist); the closest analogue is ensuring the model doesn't systematically mis-price listings in any one borough, checked implicitly via the borough-level EDA matching the model's own geospatial feature importance
- **Explainability:** `room_type_Entire home/apt` dominates (0.502), with `longitude`/`latitude` together contributing over a quarter of total importance — confirming the geospatial signal EDA found is genuinely learnable, not just a descriptive correlation

---

## Phase 10: MLOps & Deployment
**Owner:** ML Engineer | **Output:** `app.py`

- **Packaging:** FastAPI service (`/price`, `/health`, browser test form)
- **THE key design decision:** the API returns a PRICE RANGE, not a single point estimate — a deliberate departure from Day 17's single-number pricing, justified directly by this model's honestly-measured 0.16 R²: presenting one confident-looking number would overstate the model's actual precision
- **Verified with 2 constructed listings:** a Manhattan entire apartment suggested $268.75 ($211–$326 range); a Bronx shared room suggested $39.95 ($0–$98 range) — both sensible, with appropriately wide uncertainty

---

## Phase 11: Testing
**Owner:** QA / Data Scientist | **Output:** manual test log

- **What was actually tested:** full pipeline run end-to-end; API tested via FastAPI's `TestClient` with two constructed listings spanning the price spectrum, both producing sensible ranges
- **What was NOT done:** no formal unit test suite; no UAT with actual hosts; no text-mining of the `name` field, which could plausibly carry additional price signal — a stated scope simplification, not an oversight

---

## Phase 12: Documentation & Handover
**Owner:** Data Scientist | **Output:** `README.md`, `PROJECT_DOCUMENTATION.md`, this file

- `README.md` leads with a direct comparison table against Day 17, since the CONTRAST between the two pricing projects (why one achieves R²=0.94 and the other R²=0.16) is more instructive than either project read in isolation
- The counter-intuitive reviews-vs-price finding and the "not actually a data quality issue" missingness finding are both documented prominently, continuing this series' practice of surfacing genuine analytical judgment

---

## Phase 13: Monitoring & Maintenance
**Owner:** MLOps / Data Scientist | **Output:** `monitor.py`

- **What's monitored:** PSI drift on availability/reviews/minimum-nights, and MAE decay
- **Recommended cadence: monthly-to-quarterly for gradual demand drift** (same category as Day 17) **PLUS a regulatory-shock trigger unique to this project**: NYC's short-term-rental rules changed materially after this 2019 snapshot, and a policy change can shift this market discontinuously, not gradually — a genuinely different monitoring consideration than any prior day's gradual-drift-only framing

---

## Phase 14: Project Closure & Delivery
**Owner:** N/A (self-directed project) | **Output:** this document, `PROJECT_DOCUMENTATION.md`

- **Closure against original objective:** a working pricing model was built and honestly evaluated, with 43.2% of predictions landing within a realistic ±20% tolerance, and an explicit, advance-stated explanation for why this project's R² is genuinely lower than Day 17's — a methodological maturity point (correctly predicting a limitation before hitting it) rather than a disappointing result to explain away
- **Retrospective:**
  - What went well: reasoning through what this specific dataset could and couldn't observe about rental pricing BEFORE training, and setting the tolerance band and R² expectations accordingly, meant the actual results confirmed a stated hypothesis rather than requiring a defensive explanation afterward
  - What to improve next time: mine the `name` field for text signal (keywords like "luxury," "cozy," "doorman"); incorporate a more recent dataset to check whether post-regulation-change pricing dynamics differ from this 2019 snapshot
- **Handover:** packaged as a complete, runnable project (code + real data + trained model + documentation), consistent with every prior day in the series

---

## Mapping back to the Day 19 LinkedIn post

| LinkedIn section | Pulled from phases | Real number used |
|---|---|---|
| Client & Problem | 1–2 | PropTech, inconsistent host self-pricing |
| Requirements & Data | 3, 5 | Real NYC Airbnb dataset, 3 data quality findings (one not really an issue) |
| Approach | 6–8 | Borough/room-type EDA → Random Forest, expected-in-advance low R² |
| Result & Business Impact | 9 | 43.2% of predictions within ±20%; R²=0.16 vs. Day 17's 0.94, explained |
| Path to Production | 10–13 | Price-range API + market-drift + regulatory-shock monitoring |
