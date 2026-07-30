# DemandMind

**DemandMind** is a machine learning project that predicts the **weekly sales** of an electronics retail store. Since predictions are weekly, monthly sales can simply be estimated by summing 4 consecutive weekly predictions.

The project ships with data that's already cleaned, plus the final feature-engineering steps needed before feeding it into the models. Two models are trained and compared in the main notebook:

- **LinearRegression** — found in the **2nd cell** of the main notebook.
- **XGBoostRegressor** — found in the **3rd cell** of the main notebook.

---

## Why two models? Are they different?

I trained two models to compare and pick the best one (I actually tried two more models early on, but dropped them to keep the final code clean).

For hyperparameter tuning, I combined **RandomizedSearchCV** and **HalvingGridSearchCV** — an approach from *Hands-On Machine Learning* (2025 edition): use `RandomizedSearchCV` first to narrow down the promising region of the parameter space, then refine within that region using a grid search. I used the **Halving** variant of grid search specifically because it progressively eliminates weak candidates and concentrates resources on the best ones, which converges to a strong estimator more efficiently than an exhaustive grid search.

The two models are conceptually very different, even though they're used the same way in this project:
- **LinearRegression** fits a straight line (a linear combination of the features) that the data is assumed to be distributed around.
- **XGBoostRegressor** builds an ensemble of decision trees, where predictions come from combining the outputs of many trees.

## What's the best model, based on this data and these results?

Neither model exceeded **~66% R²** on the (chronological) test set, and on that headline number they're close. But looking past R² alone, **LinearRegression edges out XGBoostRegressor** on this dataset:

- Across RMSE, MAE, and MSE on the test set, LinearRegression is slightly better than XGBoostRegressor.
- LinearRegression is dramatically cheaper to train — a couple of seconds, versus roughly 2 minutes for XGBoostRegressor (which runs both `RandomizedSearchCV` and `HalvingGridSearchCV` for tuning).
- XGBoostRegressor does come out ahead on cross-validated R² specifically.

So while it's close, LinearRegression is arguably the more practical choice here: comparable (in some metrics, better) accuracy at a fraction of the training cost. Full metrics for both models are in the notebook.

### A note on cross-validation

Early on, cross-validation (a plain, non-time-aware `KFold`-style split) produced a **negative R²** for LinearRegression on some folds — meaning it performed worse than simply predicting the mean. The cause was that this CV split data **randomly rather than chronologically**, so some folds ended up training on later weeks and evaluating on earlier ones (or vice versa), breaking the time-dependent structure the model relies on (e.g. the week/day features).

Once evaluated in a way that respects time order — training on earlier periods, evaluating on later ones — results were consistent and sensible again, including RMSE staying appropriately larger than MAE (expected, since squaring penalizes large errors more).

**Known limitation:** the CV setup used during experimentation was not time-aware. A more rigorous version of this project would use `TimeSeriesSplit` (or an equivalent rolling/expanding-window CV) throughout, rather than relying on a single chronological train/test split. This is a good next step for anyone building on this project.

I also ran a quick exploratory test late in development: generating a new, synthetic dataset (substantially different from the original data) with a clean chronological structure, and scoring the trained model on it — which returned a higher accuracy (~82–85%). This is **not** treated as a reliable generalization metric here, since synthetic data is typically simpler/less noisy than real data and isn't necessarily representative of new real-world samples the way the original test set is. It's included as a side note, not as evidence of model performance — the **~62–66% test R² above is the trustworthy number**.

## Data preparation notes

A few things worth knowing about how the data was prepared before training:

- **This is time-series data, treated as such.** Two engineered columns — week and day — were added, and the **train/test split is chronological**, not random. The model is trained on earlier periods and evaluated on later ones, so it never trains on the "future" to predict the "past."
- **I initially tried a random/stratified split** to see if it would help. It didn't — it actually made things worse (test accuracy dropped to ~8%, down from ~13–14% with a proper time-based split). This makes sense in hindsight: shuffling time-series data before splitting lets information from later weeks leak into training, so the resulting metrics aren't just lower, they're not trustworthy in the first place. This confirmed that a chronological split was the correct approach for this problem, and it's what the final version uses.
- **Log transformation** was applied to skewed numeric features — notably price, which ranges from as low as **2.49** up to **1700**, with most values clustered around **400–1000**. This right-skewed distribution can disproportionately influence the model (especially LinearRegression), so a log transform was used to compress the scale and reduce the impact of extreme values.
- **Categorical encoding:** `OneHotEncoder` was used for categorical features.
- Because the data is time-ordered, it was **not shuffled** at any point before splitting.

### Before vs. after these adjustments

| | Before tuning/preprocessing | After |
|---|---|---|
| Train accuracy | ~14% | ~80–82% |
| Test accuracy | ~13% | ~62–66% |

All other metrics (RMSE, MAE, MSE) are shown in the notebook.

## Project structure

- **`Core.ipynb`** ([Python](./Python)) — the main notebook. Loads the data (path is set to my local Kaggle path — update it to your own), applies the final preprocessing steps described above, and trains both models.
- **`SalesModel.ipynb`** ([Python](./Python)) — the notebook that performs the initial data cleaning, from the raw data onward.
- **`Model/`** — contains the final, fully processed dataset (ready to feed directly into the model) and the trained model itself.

## Evaluation metrics

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
