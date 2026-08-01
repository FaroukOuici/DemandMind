# DemandMind

**DemandMind** predicts **weekly sales** (quantity ordered) for an electronics retail store, broken down by product. Since predictions are weekly, monthly sales can be estimated by summing 4 consecutive weekly predictions.

The project repository includes the cleaned, feature-engineered dataset along with the trained model. Two models are trained, evaluated, and compared in the main notebook:
1. **LinearRegression**
2. **XGBoostRegressor**

---

## 🛠️ Data Preparation & Feature Engineering

Sales data is inherently time-dependent, so every choice below exists to solve a specific problem that comes from that fact — not as a generic "best practice" checklist.

* **Chronological splitting — problem: future leakage.** A random or stratified split would let the model train on some weeks that come *after* the weeks it's tested on, which lets it implicitly "see the future" and produces test scores that look good but don't reflect real predictive ability. The dataset is instead split strictly by time — training on earlier weeks, testing on later ones — with no shuffling anywhere in the pipeline.
* **A separate holdout set (`df_test`) — problem: needed an extra, harder guarantee against leakage.** Before the regular train/test split even happens, the most recent 10% of the data is carved off and set aside completely. The model never sees it during training, tuning, or the regular test evaluation — it's only used once, at the very end, to check performance on truly unseen future weeks.
* **Lag & rolling features (`Qty_Lag_1`, `Qty_Rolling_3`) — problem: the model had no memory of recent demand.** Without these, the model only sees a snapshot (price, product, week) with no sense of what that product's demand was just doing. `Qty_Lag_1` gives it last period's quantity per product; `Qty_Rolling_3` gives it a 3-period rolling average per product to smooth out short-term noise. These turned out to matter a lot — they're a large part of why performance is as strong as it is.
* **Cyclical week encoding (`Week_sin`, `Week_cos`) — problem: week numbers don't wrap around correctly.** Treating `Week` as a plain number tells the model week 52 and week 1 are as far apart as possible, when they're actually adjacent (one year rolling into the next). Sine/cosine transforms encode the week on a circle instead of a line, so the model sees that wraparound correctly.
* **Log transformation of `Price_Each` — problem: extreme price outliers skewing the model.** Prices range from about 2.5 up to ~1700, with most values clustered around 400–1000 — a heavily right-skewed distribution that can disproportionately pull a model (especially LinearRegression) toward fitting the extremes. A `log1p` transform compresses that scale before the values are standardized with `StandardScaler`.
* **Log transformation of the target (`Quantity_Ordered`) — same problem, applied to the label.** Demand itself spikes unevenly across products and periods. The target is log-transformed during training via `TransformedTargetRegressor` (`log1p` in, `expm1` out), so the model trains on a more stable scale and predictions are automatically converted back to real quantities.
* **`OneHotEncoder` for `Product`** — standard categorical encoding, since product is a category with no inherent order.

---

## 🤖 Model Optimization & Tuning

For **XGBoostRegressor**, hyperparameter tuning is a two-stage search, wrapped inside the same `TransformedTargetRegressor` used for the target log-transform:
1. **`RandomizedSearchCV`** broadly explores the hyperparameter space (tree depth, learning rate, subsample/colsample ratios) using `TimeSeriesSplit` cross-validation, to cheaply locate a promising region without an exhaustive search.
2. **`GridSearchCV`** then fine-tunes around the best parameters found by the randomized search — a small, focused grid rather than a large one, since the search space has already been narrowed.

This combination — random search to find the right neighborhood, then grid search to refine within it — follows the approach described in *Hands-On Machine Learning*.

---

## 📊 Cross-Validation & Evaluation Strategy

Problem this section solves: an earlier version of this project used standard random `KFold` cross-validation, which — for the same future-leakage reason as above — produced **negative R² scores**: some folds ended up training on later weeks and evaluating on earlier ones, breaking the time-dependent structure the model relies on. Replacing it with **`TimeSeriesSplit` (`n_splits=5`)** for *every* CV call, for both models, fixed this: every fold now trains strictly on a historical slice and evaluates on a subsequent one.

* Metrics tracked: R², RMSE, and MAE — on cross-validation, on the regular train/test split, and separately on the `df_test` holdout.
* RMSE is consistently ≥ MAE across results, as expected mathematically (squaring penalizes large errors more than the absolute-value term does) — a useful sanity check that the numbers are internally consistent.

---

## 📈 Model Performance & Comparison

| Metric | LinearRegression | XGBoostRegressor |
| :--- | :---: | :---: |
| Train R² | 0.9353 | 0.9735 |
| Test R² | 0.8054 | 0.7263 |
| Mean CV R² (TimeSeriesSplit) | 0.9330 | 0.9531 |
| **Holdout R² (`df_test`, unseen future weeks)** | **0.8768** | **0.3802** |
| Holdout MAE | 24.15 | 51.77 |
| Holdout RMSE | 32.49 | 72.88 |
| Training time | ~0.02 s | ~10 s (random + grid search) |

### Key takeaway: LinearRegression is the better model here

On the regular test split and on cross-validation, XGBoost looks slightly ahead of LinearRegression. But on `df_test` — real, chronologically later data the model never touched in any form during training or tuning — **XGBoost's R² collapses from 0.73 to 0.38**, while LinearRegression's *improves* to 0.88.

Read together with XGBoost's higher train R² (0.97 vs. 0.94), this is a textbook overfitting signature: XGBoost is picking up fine-grained patterns specific to the training period that don't hold up on new time periods, while the simpler linear model generalizes more reliably. Combined with being roughly 500x faster to train, **LinearRegression is the practical choice for this problem** — the saved model (`DemandMind.pkl`) is the LinearRegression pipeline.

---

## 📁 Repository Structure

* `demandmind.ipynb` — the main notebook: data loading, time-based feature engineering, chronological + holdout splitting, both model pipelines, hyperparameter tuning, and full evaluation.
* `electronic-store-sales_cleaned.csv` — the cleaned, feature-engineered dataset (excludes the final holdout slice), ready to feed into the models.
* `DemandMind.pkl` — the saved LinearRegression pipeline (via `joblib`), ready for inference.

---

## 📐 Evaluation Metrics

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
