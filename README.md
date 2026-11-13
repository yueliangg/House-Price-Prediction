# House Price Prediction (Regression)

Predicting house sale prices from property features such as size, layout, city and renovation status, and identifying which factors drive price the most.

> **Final model:** Optuna-tuned Ridge Regression on the top 10 features. RMSE ≈ **$152,482**, MAE ≈ **$113,384**, R² ≈ **0.540**, using 10 features instead of 18.

---

## Problem Overview

| | |
|---|---|
| **Task** | Regression |
| **Dataset** | `housing_price_data.csv`: 545 houses, 8 columns |
| **Target** | `Price ($)`, ranging from $175,000 to $1,330,000 |
| **Features** | House area, bedrooms, toilets, storeys, city, renovation status |

The price distribution is **right-skewed** (skewness 1.21): most homes cost $250k–$650k, with a small number above $1M.

---

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `Optuna` · `Matplotlib` · `Seaborn` · `Jupyter`

---

##  Workflow

1. **Exploratory Data Analysis:** price distribution, correlation heatmap, and price by area, bedrooms, toilets, city and renovation status.
2. **Data Cleaning:** no missing values or duplicates; dropped the `House ID` identifier.
3. **Encoding:**
   - `Renovation Status` → ordinal (unfurnished = 0, semi-furnished = 1, furnished = 2).
   - `City` → one-hot encoded (`drop_first=True`).
4. **Feature Engineering:** 9 new features:
   - **Ratios:** Total Rooms, Area per Room, Baths per Bedroom
   - **Interactions:** Bedrooms × Renovation, Toilets × Renovation, Bedrooms × Stories, Area per Room × Renovation
   - **Non-linear terms:** HouseArea², Stories²
5. **Modelling:** Dummy baseline, Linear Regression, Ridge, Random Forest and Gradient Boosting, on an 80/20 split with `StandardScaler` fitted on the training set only.
6. **Hyperparameter Tuning:** Ridge `alpha` and `fit_intercept` tuned with `Optuna`.
7. **Interpretation:** standardised Ridge coefficients, including whether each feature raises or lowers price.
8. **Improvement:** kept only the top 10 features and re-tuned the model.

---

##  Results

Test set: 109 houses.

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Dummy (mean) | 226,830 | 174,862 | −0.018 |
| Gradient Boosting | 164,193 | 120,241 | 0.467 |
| Random Forest | 162,054 | 115,294 | 0.480 |
| Ridge (Base, 18 features) | 153,221 | 114,102 | 0.536 |
| Linear Regression | 152,885 | 113,736 | 0.538 |
| Ridge (Optuna, 18 features) | 152,885 | 113,736 | 0.538 |
| Ridge (Base, Top 10 features) | 152,688 | 113,647 | 0.539 |
| **Ridge (Optuna, Top 10 features)** | **152,482** | **113,384** | **0.540** |

- **Every model beats the dummy baseline.** The best model cuts RMSE by about $74,000.
- **Linear models beat the tree-based ensembles.** The engineered features already captured the non-linear patterns, and the dataset is small.
- **Feature trimming and tuning each gave a small improvement,** and combining them gave the best and simplest model.

---

##  Key Insights

- **House Area** (r = 0.54) and **No. of Toilets** (r = 0.52) are the strongest predictors of price.
- **Diminishing returns on size.** HouseArea² has a negative coefficient, so price grows more slowly for very large houses. The most expensive homes are mid-sized (740–1,000 sqm), not the largest.
- **Renovation matters mainly through interactions.** Furnishing adds the most value to houses with more bedrooms.
- **City has very little effect** on price in this dataset.

---

##  How to Run

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install pandas numpy matplotlib seaborn scikit-learn optuna
jupyter notebook p2534945_CA1_PartB_Thawdar_Lin_Wai.ipynb
```

In the notebook, set the dataset path in the loading cell:

```python
df = pd.read_csv("data/housing_price_data.csv")
```

Then select **Kernel → Restart & Run All**. It runs in under a minute, and results are reproducible (`random_state=42`, seeded Optuna sampler).

---

## Future Improvements

- **Predict `log(Price)`** to reduce the effect of the few houses above $1M, which the model currently under-predicts.
- **Tune with k-fold cross-validation** on the training set, so the test set stays fully unseen until the final evaluation.
- **Add features such as property age, neighbourhood and amenities.** About half of the price variation (R² ≈ 0.54) is still unexplained.

---

## Author

**Thawdar Lin Wai**
Diploma student, Singapore Polytechnic
