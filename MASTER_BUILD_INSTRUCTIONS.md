# MASTER BUILD INSTRUCTIONS — Capstone End-to-End

**You are Claude Code. Build this entire capstone project for Tinka Fahad. Do everything yourself. Do not stop to ask the user questions unless absolutely necessary (Kaggle credentials, GitHub username). When you hit a decision point and there's a reasonable default, take it.**

---

## Project metadata

- **Authors:** Tinka Fahad (Student ID 254430) and Tugume Andrew
- **Institution:** Cavendish University Uganda
- **Programme:** BSc Data Science & AI
- **Course:** Predictive Analytics Capstone
- **Date:** May 2026
- **Project title:** Predicting Used Vehicle Prices in an Emerging Market: A Regression Analysis of Indian Used-Car Listings with Application to Uganda's Ride-Hailing Driver Vehicle Acquisition

---

## Your mission, in order

1. Set up the project folder and Python virtual environment
2. Help the user download the CarDekho dataset from Kaggle
3. Build the complete Jupyter notebook (12 sections, all three regression models)
4. Run the notebook end-to-end and verify it works
5. Build the Streamlit web application
6. Test that the app launches and predicts correctly
7. Generate the Word report
8. Set up git, write the README and .gitignore, create the GitHub repository, push everything
9. Print a final summary with all the commands the user needs for the video demo

**At the end, the user should have everything ready to record their video and submit. Do not stop halfway. Do not ask the user to do things you can do yourself.**

---

## PHASE 1 — Project setup

Create the project folder, set up Python, install everything.

```bash
mkdir -p ~/capstone-vehicle-prices
cd ~/capstone-vehicle-prices

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook ipykernel kaggle streamlit joblib python-docx nbformat

python -m ipykernel install --user --name=capstone --display-name "Capstone (Python)"
```

Verify everything installed:

```bash
python -c "import pandas, numpy, sklearn, streamlit, joblib, docx; print('All imports OK')"
```

---

## PHASE 2 — Download the dataset

Try the Kaggle API first. If the user doesn't have it configured, instruct them clearly.

```bash
# Check if Kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Kaggle API not configured."
    echo ""
    echo "TINKA: Please do the following ONE TIME setup:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Scroll to 'API' section, click 'Create New Token'"
    echo "3. This downloads kaggle.json"
    echo "4. Move it to ~/.kaggle/kaggle.json"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "Then tell me you're ready and I'll continue."
    exit 1
fi

# Download the dataset
kaggle datasets download -d nehalbirla/vehicle-dataset-from-cardekho
unzip -o vehicle-dataset-from-cardekho.zip
ls -la "Car details v3.csv"
```

If the Kaggle CLI download fails for any reason (network, credentials), STOP and tell the user:

> "Kaggle download failed. Please download manually from https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho and place `Car details v3.csv` in `~/capstone-vehicle-prices/`. Tell me when done."

Then wait for the user before continuing.

After the file is in place, verify:

```bash
head -1 "Car details v3.csv"
wc -l "Car details v3.csv"
```

Expected: 8,128 rows + 1 header, column names including `name,year,selling_price,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,torque,seats`.

---

## PHASE 3 — Build the Jupyter notebook

Create `capstone_vehicle_prices.ipynb` using Python's `nbformat` library. Write a script `build_notebook.py` at the project root that constructs the notebook programmatically. Then run it.

The notebook MUST contain the following sections in order. Use markdown cells for headers and code cells for code. Cells are defined exactly as below — do not paraphrase or skip.

### Notebook content specification

**Title block (markdown cell):**

```markdown
# Predicting Used Vehicle Prices in an Emerging Market

**A Regression Analysis of Indian Used-Car Listings with Application to Uganda's Ride-Hailing Driver Vehicle Acquisition**

---

**Authors:**
- **Tinka Fahad** — Student ID: 254430
- **Tugume Andrew**

**Programme:** BSc Data Science & Artificial Intelligence
**Institution:** Cavendish University Uganda
**Course:** Predictive Analytics Capstone
**Date:** May 2026

---

## Abstract

This study develops and compares three regression models — Simple Linear Regression, Polynomial Regression, and Random Forest Regression — to predict the resale price of used vehicles, using the Indian CarDekho dataset (8,128 listings). After cleaning, the final modeling dataset contained approximately 7,900 records spanning vehicles from 1994 to 2020. Random Forest Regression achieved the strongest predictive performance, substantially outperforming the linear baseline. The dataset was selected because the Indian used-vehicle market is structurally analogous to Uganda's: both are right-hand-drive markets dominated by Japanese manufacturers (Maruti-Suzuki, Toyota, Honda, Hyundai) with strong import-driven economics. Findings are therefore directly applicable to Uganda's ride-hailing driver community, where vehicle acquisition is the largest capital decision a driver makes. The lead author is the founder of TinkaTaxi, a Kampala-based ride-hailing platform, and an active driver on competing platforms — domain experience that grounds this work in real operational context.

## Motivation and Domain Context

As founder of TinkaTaxi, a Kampala-based ride-hailing platform serving over 3,000 users and 2,000 drivers, and as an active driver on competing platforms, the lead author has observed first-hand that vehicle acquisition is the single largest capital decision a ride-hailing driver makes in the Ugandan market. Drivers routinely overpay for unreliable vehicles or undervalue serviceable ones because no transparent, data-driven pricing reference exists for used vehicles in this market.

Uganda's used-vehicle market is overwhelmingly composed of Japanese right-hand-drive imports — Toyota Premio, Toyota Wish, Honda Fit, Nissan Note, Suzuki Alto. Direct Ugandan data of sufficient scale is not publicly available. India's used-car market is the closest structural analogue: it is right-hand-drive, dominated by Japanese manufacturers (especially through Maruti-Suzuki), and shaped by the same import-and-resale dynamics that govern Uganda's fleet. The CarDekho dataset, which captures listings from India's largest used-car marketplace, therefore provides the most relevant publicly available training data for a Ugandan price-prediction methodology.

## Marking Scheme Mapping

| Rubric Component | Marks | Notebook Section |
|---|---|---|
| Data Preparation & Preprocessing | 20 | Sections 2–4 |
| Data Understanding & Exploration | 20 | Sections 5–7 |
| Regression Model 1 (Linear) | 20 | Section 8 |
| Regression Model 2 (Polynomial) | 20 | Section 9 |
| Regression Model 3 (Random Forest) + Conclusion | 20 | Sections 10–12 |
| **Total** | **100** | |
```

**Section 2 — Imports (markdown + code):**

Header: `## 2. Environment Setup and Library Imports`

Code:
```python
import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("All libraries loaded successfully.")
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
```

**Section 3 — Load data (markdown + code × 2):**

Header: `## 3. Data Loading and Initial Inspection`

Code 1 — load:
```python
df_raw = pd.read_csv('Car details v3.csv')
print(f"Raw dataset shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"\nColumn names:")
print(df_raw.columns.tolist())
df_raw.head(5)
```

Code 2 — inspect:
```python
summary = pd.DataFrame({
    'dtype': df_raw.dtypes,
    'non_null_count': df_raw.notna().sum(),
    'null_count': df_raw.isna().sum(),
    'null_pct': (df_raw.isna().sum() / len(df_raw) * 100).round(2)
})
summary.sort_values('null_pct', ascending=False)
```

**Section 4 — Data Preparation & Preprocessing (20 marks):**

Header markdown:
```markdown
## 4. Data Preparation & Preprocessing (20 marks)

This section corresponds to the first rubric component. We perform the following operations:

1. **Extract numeric values** from mixed text/numeric columns (`mileage` contains "23.4 kmpl", `engine` contains "1248 CC", `max_power` contains "74 bhp").
2. **Extract the manufacturer** from the `name` column.
3. **Handle missing values** — drop rows missing critical predictors.
4. **Filter implausible records** — remove pricing/usage outliers.
5. **Feature engineering** — derive `age` from `year`, log-transform skewed variables.
```

Code cell 1 — numeric extraction:
```python
df = df_raw.copy()

def extract_first_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r'[\d.]+', str(value))
    return float(match.group()) if match else np.nan

df['mileage_kmpl'] = df['mileage'].apply(extract_first_number)
df['engine_cc'] = df['engine'].apply(extract_first_number)
df['max_power_bhp'] = df['max_power'].apply(extract_first_number)
df['manufacturer'] = df['name'].str.split().str[0]

print("Numeric extraction complete.")
print(df[['name', 'manufacturer', 'mileage_kmpl', 'engine_cc', 'max_power_bhp']].head())
```

Code cell 2 — drop original mixed columns:
```python
columns_to_drop = ['name', 'mileage', 'engine', 'max_power', 'torque']
df = df.drop(columns=columns_to_drop)
print(f"After dropping mixed-format columns: {df.shape[1]} columns remain")
print(f"Remaining columns: {df.columns.tolist()}")
```

Code cell 3 — handle missing:
```python
print(f"Before dropping missing critical values: {len(df):,} rows")
critical_columns = ['mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
df = df.dropna(subset=critical_columns)
print(f"After: {len(df):,} rows")
```

Code cell 4 — outlier filtering:
```python
print(f"Before outlier filtering: {len(df):,} rows")
df = df[(df['selling_price'] >= 50_000) & (df['selling_price'] <= 10_000_000)]
df = df[(df['km_driven'] >= 0) & (df['km_driven'] <= 500_000)]
df = df[(df['year'] >= 1990) & (df['year'] <= 2021)]
print(f"After: {len(df):,} rows")
print(f"Final price range: INR {df['selling_price'].min():,} to INR {df['selling_price'].max():,}")
```

Code cell 5 — feature engineering:
```python
df['age'] = 2021 - df['year']
df['log_km_driven'] = np.log1p(df['km_driven'])
df['log_selling_price'] = np.log(df['selling_price'])

INR_TO_UGX = 45
df['selling_price_ugx_approx'] = df['selling_price'] * INR_TO_UGX

print(f"Final cleaned dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Reduction from raw: {(1 - len(df) / len(df_raw)) * 100:.1f}% of rows removed")

df.to_csv('vehicles_clean.csv', index=False)
print("Saved vehicles_clean.csv")
```

**Section 5 — Descriptive Stats (markdown + code × 2):**

Header: `## 5. Data Understanding & Exploration (20 marks) — Descriptive Statistics`

Code 1:
```python
numeric_features = ['selling_price', 'km_driven', 'age', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
desc = df[numeric_features].describe().round(2)
desc.loc['mode'] = df[numeric_features].mode().iloc[0]
desc.loc['median'] = df[numeric_features].median()
desc = desc.round(2)
desc
```

Code 2:
```python
categorical_features = ['manufacturer', 'fuel', 'transmission', 'seller_type', 'owner']
for col in categorical_features:
    print(f"\n{col.upper()} — Top 10 categories:")
    print(df[col].value_counts().head(10))
    print(f"Total unique values: {df[col].nunique()}")
```

**Section 6 — Visualisations:**

Header markdown: `## 6. Data Understanding & Exploration — Visualisations`

Code cell 1 — price histogram:
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['selling_price'] / 100_000, bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Selling Price (raw)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Selling Price (INR Lakhs)')
axes[0].set_ylabel('Count')
median_price = df['selling_price'].median() / 100_000
axes[0].axvline(median_price, color='red', linestyle='--', label=f'Median = INR {median_price:.1f} Lakh')
axes[0].legend()

axes[1].hist(df['log_selling_price'], bins=50, color='darkorange', edgecolor='white')
axes[1].set_title('Distribution of log(Selling Price)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(Selling Price)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('fig_price_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
```

Code cell 2 — age + km histograms:
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['age'], bins=30, color='seagreen', edgecolor='white')
axes[0].set_title('Distribution of Vehicle Age (years)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')

axes[1].hist(df['km_driven'] / 1000, bins=50, color='indianred', edgecolor='white')
axes[1].set_title('Distribution of Kilometres Driven', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Kilometres Driven (thousands)')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('fig_age_km_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
```

Code cell 3 — fuel pie:
```python
fuel_counts = df['fuel'].value_counts()
fig, ax = plt.subplots(figsize=(9, 9))
colors = sns.color_palette('Set2', len(fuel_counts))
ax.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%',
       colors=colors, startangle=90, textprops={'fontsize': 12})
ax.set_title('Vehicle Distribution by Fuel Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_fuel_pie.png', dpi=120, bbox_inches='tight')
plt.show()
```

Code cell 4 — manufacturers bar:
```python
top_manufacturers = df['manufacturer'].value_counts().head(15)
fig, ax = plt.subplots(figsize=(12, 6))
top_manufacturers.plot(kind='bar', color='steelblue', edgecolor='white', ax=ax)
ax.set_title('Top 15 Vehicle Manufacturers (CarDekho India)', fontsize=13, fontweight='bold')
ax.set_xlabel('Manufacturer')
ax.set_ylabel('Number of Listings')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('fig_top_manufacturers.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Section 7 — Correlation analysis:**

Header markdown: `## 7. Correlation Analysis (rubric requirement)`

Code cell 1 — heatmap:
```python
corr_features = ['selling_price', 'log_selling_price', 'year', 'age', 'km_driven',
                 'log_km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
corr_matrix = df[corr_features].corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8}, ax=ax)
ax.set_title('Correlation Matrix — Numeric Features', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('fig_correlation_heatmap.png', dpi=120, bbox_inches='tight')
plt.show()

print(f"Price vs Age: {corr_matrix.loc['selling_price', 'age']:.3f}")
print(f"Price vs km_driven: {corr_matrix.loc['selling_price', 'km_driven']:.3f}")
print(f"Price vs max_power_bhp: {corr_matrix.loc['selling_price', 'max_power_bhp']:.3f}")
print(f"Price vs engine_cc: {corr_matrix.loc['selling_price', 'engine_cc']:.3f}")
```

Code cell 2 — scatter plots:
```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample = df.sample(n=min(3000, len(df)), random_state=RANDOM_STATE)

axes[0].scatter(sample['age'], sample['selling_price'] / 100_000, alpha=0.3, s=12, color='steelblue')
axes[0].set_title('Price vs Age', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Price (INR Lakhs)')

axes[1].scatter(sample['km_driven'] / 1000, sample['selling_price'] / 100_000, alpha=0.3, s=12, color='indianred')
axes[1].set_title('Price vs Kilometres Driven', fontsize=13, fontweight='bold')
axes[1].set_xlabel('km driven (thousands)')
axes[1].set_ylabel('Price (INR Lakhs)')

plt.tight_layout()
plt.savefig('fig_scatter_relationships.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Section 8 — Model 1 Simple Linear (20 marks):**

Header markdown:
```markdown
## 8. Regression Model 1 — Simple Linear Regression (20 marks)

We establish a baseline using Simple Linear Regression with vehicle age as the single predictor. This is the simplest possible regression — the most intuitive baseline against which more sophisticated models will be compared.

**Hypothesis:** Vehicle price decreases approximately linearly with age.

**Expected outcome:** This model will perform poorly because the true relationship is non-linear (depreciation follows an exponential curve, not a straight line). The poor performance here motivates Sections 9 and 10.
```

Code cells:
```python
X_simple = df[['age']].copy()
y = df['selling_price'].copy()

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Training set: {len(X_train_s):,} rows")
print(f"Test set: {len(X_test_s):,} rows")
```

```python
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train)
y_pred_s = model_simple.predict(X_test_s)

rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_s))
mae_s = mean_absolute_error(y_test, y_pred_s)
r2_s = r2_score(y_test, y_pred_s)

print("=" * 60)
print("MODEL 1: Simple Linear Regression (Price ~ Age)")
print("=" * 60)
print(f"Coefficient: INR {model_simple.coef_[0]:,.2f} per year of age")
print(f"Intercept: INR {model_simple.intercept_:,.2f}")
print(f"RMSE: INR {rmse_s:,.2f}")
print(f"MAE:  INR {mae_s:,.2f}")
print(f"R²:   {r2_s:.4f}")
```

```python
fig, ax = plt.subplots(figsize=(11, 6))
sample_idx = np.random.choice(len(X_test_s), size=min(2000, len(X_test_s)), replace=False)
ax.scatter(X_test_s.iloc[sample_idx]['age'], y_test.iloc[sample_idx] / 100_000,
           alpha=0.25, s=12, color='steelblue', label='Actual prices')

age_range = np.linspace(0, df['age'].max(), 100).reshape(-1, 1)
predicted_line = model_simple.predict(age_range) / 100_000
ax.plot(age_range, predicted_line, color='red', linewidth=2.5, label='Linear regression fit')

ax.set_title('Simple Linear Regression: Price ~ Age', fontsize=14, fontweight='bold')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Price (INR Lakhs)')
ax.legend()
plt.tight_layout()
plt.savefig('fig_model1_linear.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Section 9 — Model 2 Polynomial (20 marks):**

Header markdown:
```markdown
## 9. Regression Model 2 — Polynomial Regression (20 marks)

To capture the curvature observed in Section 8, we fit polynomial regression of degree 2, using four numeric predictors: age, km_driven, max_power_bhp, engine_cc. This combines two improvements over Model 1:

1. **More predictors** (multivariate)
2. **Non-linear feature transformations** (quadratic terms)

**Hypothesis:** Polynomial features will capture depreciation curvature and outperform the linear baseline substantially.
```

Code:
```python
poly_features = ['age', 'km_driven', 'max_power_bhp', 'engine_cc']
X_poly = df[poly_features].copy()

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_poly, y, test_size=0.2, random_state=RANDOM_STATE
)

poly_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', LinearRegression())
])

poly_pipeline.fit(X_train_p, y_train_p)
y_pred_p = poly_pipeline.predict(X_test_p)

rmse_p = np.sqrt(mean_squared_error(y_test_p, y_pred_p))
mae_p = mean_absolute_error(y_test_p, y_pred_p)
r2_p = r2_score(y_test_p, y_pred_p)

print("=" * 60)
print("MODEL 2: Polynomial Regression (degree=2)")
print(f"Features: {poly_features}")
print("=" * 60)
print(f"RMSE: INR {rmse_p:,.2f}")
print(f"MAE:  INR {mae_p:,.2f}")
print(f"R²:   {r2_p:.4f}")
print(f"\nImprovement over Model 1 (R²): {r2_p - r2_s:+.4f}")
```

```python
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(poly_pipeline, X_poly, y, cv=kf, scoring='r2')
print(f"5-fold cross-validation R² scores: {cv_scores.round(4)}")
print(f"Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

```python
fig, ax = plt.subplots(figsize=(11, 6))
sample_idx = np.random.choice(len(X_test_p), size=min(2000, len(X_test_p)), replace=False)
ax.scatter(X_test_p.iloc[sample_idx]['age'], y_test_p.iloc[sample_idx] / 100_000,
           alpha=0.25, s=12, color='steelblue', label='Actual prices')

age_range = np.linspace(0, df['age'].max(), 100)
pred_input = pd.DataFrame({
    'age': age_range,
    'km_driven': [df['km_driven'].median()] * 100,
    'max_power_bhp': [df['max_power_bhp'].median()] * 100,
    'engine_cc': [df['engine_cc'].median()] * 100,
})
predicted_curve = poly_pipeline.predict(pred_input) / 100_000
ax.plot(age_range, predicted_curve, color='darkorange', linewidth=2.5, label='Polynomial fit (deg=2)')

ax.set_title('Polynomial Regression', fontsize=14, fontweight='bold')
ax.set_xlabel('Age (years)')
ax.set_ylabel('Price (INR Lakhs)')
ax.legend()
plt.tight_layout()
plt.savefig('fig_model2_polynomial.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Section 10 — Model 3 Random Forest (20 marks):**

Header markdown:
```markdown
## 10. Regression Model 3 — Random Forest Regression (20 marks)

Previous models used only numeric features. Vehicle price, however, depends heavily on **categorical** features — manufacturer, fuel type, transmission, ownership history. Random Forest handles these natively (after encoding) and captures complex non-linear interactions.

**Hypothesis:** Random Forest will substantially outperform both prior models because it (a) handles categorical features and (b) captures interaction effects.
```

Code:
```python
rf_numeric = ['age', 'km_driven', 'mileage_kmpl', 'engine_cc', 'max_power_bhp', 'seats']
rf_categorical = ['manufacturer', 'fuel', 'transmission', 'seller_type', 'owner']

def cap_cardinality(series, top_n=15):
    top_values = series.value_counts().head(top_n).index
    return series.where(series.isin(top_values), other='other')

df_rf = df.copy()
df_rf['manufacturer'] = cap_cardinality(df_rf['manufacturer'], top_n=15)

X_rf = df_rf[rf_numeric + rf_categorical].copy()
y_rf = df_rf['selling_price'].copy()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Random Forest training set: {len(X_train_r):,} rows × {X_train_r.shape[1]} features")
```

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', rf_numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), rf_categorical)
    ]
)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200, max_depth=20, min_samples_leaf=3,
        n_jobs=-1, random_state=RANDOM_STATE
    ))
])

print("Fitting Random Forest...")
rf_pipeline.fit(X_train_r, y_train_r)

y_pred_r = rf_pipeline.predict(X_test_r)
rmse_r = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
mae_r = mean_absolute_error(y_test_r, y_pred_r)
r2_r = r2_score(y_test_r, y_pred_r)

print("=" * 60)
print("MODEL 3: Random Forest Regression")
print("=" * 60)
print(f"RMSE: INR {rmse_r:,.2f}")
print(f"MAE:  INR {mae_r:,.2f}")
print(f"R²:   {r2_r:.4f}")
```

```python
encoded_names = (rf_numeric +
                 list(rf_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(rf_categorical)))

importances = rf_pipeline.named_steps['regressor'].feature_importances_
imp_df = pd.DataFrame({'feature': encoded_names, 'importance': importances}).sort_values('importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='seagreen')
ax.set_title('Top 15 Feature Importances — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance score')
plt.tight_layout()
plt.savefig('fig_model3_feature_importance.png', dpi=120, bbox_inches='tight')
plt.show()

print("Top 5 features by importance:")
print(imp_df.head().to_string(index=False))
```

```python
fig, ax = plt.subplots(figsize=(10, 10))
sample_idx = np.random.choice(len(y_test_r), size=min(2000, len(y_test_r)), replace=False)
ax.scatter(y_test_r.iloc[sample_idx] / 100_000, y_pred_r[sample_idx] / 100_000,
           alpha=0.3, s=14, color='seagreen')

lims = [0, max(y_test_r.max(), y_pred_r.max()) / 100_000]
ax.plot(lims, lims, '--', color='red', linewidth=2, label='Perfect prediction')

ax.set_xlabel('Actual price (INR Lakhs)', fontsize=12)
ax.set_ylabel('Predicted price (INR Lakhs)', fontsize=12)
ax.set_title('Random Forest: Predicted vs Actual Prices', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('fig_model3_pred_vs_actual.png', dpi=120, bbox_inches='tight')
plt.show()
```

**Section 11 — Model comparison:**

Header markdown: `## 11. Model Comparison`

Code:
```python
results = pd.DataFrame({
    'Model': ['Simple Linear (Model 1)', 'Polynomial deg=2 (Model 2)', 'Random Forest (Model 3)'],
    'RMSE (INR)': [rmse_s, rmse_p, rmse_r],
    'MAE (INR)':  [mae_s,  mae_p,  mae_r],
    'R²':         [r2_s,   r2_p,   r2_r]
}).round(4)

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [('RMSE (INR)', 'RMSE — lower is better', 'indianred'),
           ('MAE (INR)',  'MAE — lower is better',  'darkorange'),
           ('R²',         'R² — higher is better',  'seagreen')]

for ax, (metric, title, color) in zip(axes, metrics):
    ax.bar(results['Model'], results[metric], color=color, edgecolor='white')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig('fig_model_comparison.png', dpi=120, bbox_inches='tight')
plt.show()

results.to_csv('model_comparison.csv', index=False)
```

**Section 11.5 — Save model for UI:**

Header markdown: `## 11.5 Saving the Trained Model for Deployment`

Code:
```python
import joblib
import json

joblib.dump(rf_pipeline, 'rf_pipeline.joblib')

ui_metadata = {
    'manufacturers': sorted(df_rf['manufacturer'].unique().tolist()),
    'fuel_types': sorted(df_rf['fuel'].unique().tolist()),
    'transmissions': sorted(df_rf['transmission'].unique().tolist()),
    'seller_types': sorted(df_rf['seller_type'].unique().tolist()),
    'owner_types': sorted(df_rf['owner'].unique().tolist()),
    'year_min': int(df['year'].min()),
    'year_max': int(df['year'].max()),
    'km_driven_max': int(df['km_driven'].max()),
    'engine_cc_min': int(df['engine_cc'].min()),
    'engine_cc_max': int(df['engine_cc'].max()),
    'max_power_min': float(df['max_power_bhp'].min()),
    'max_power_max': float(df['max_power_bhp'].max()),
    'mileage_min': float(df['mileage_kmpl'].min()),
    'mileage_max': float(df['mileage_kmpl'].max()),
    'seats_options': sorted([int(s) for s in df['seats'].unique()]),
    'model_metrics': {
        'rmse_inr': float(rmse_r),
        'mae_inr': float(mae_r),
        'r2': float(r2_r)
    }
}

with open('ui_metadata.json', 'w') as f:
    json.dump(ui_metadata, f, indent=2)

print("Saved rf_pipeline.joblib and ui_metadata.json")
```

**Section 12 — Conclusion (markdown only):**

```markdown
## 12. Conclusion (part of Model 3's 20 marks)

### Which model is best?

**Random Forest Regression is the best-performing model for this problem and dataset.** It achieves the lowest RMSE and MAE, and the highest R², by a substantial margin over both alternatives.

### Why does Random Forest win?

Three reasons, in order of importance:

1. **Categorical features dominate the prediction.** Manufacturer, fuel type, and ownership history carry significant predictive weight. Linear and polynomial models cannot use these directly without encoding, and even then their predictive ceiling is limited.

2. **Non-linear interaction effects.** A 10-year-old Toyota does not depreciate the same way as a 10-year-old Tata. Random Forest captures these interactions automatically.

3. **Robustness to outliers.** Tree-based models are insensitive to extreme values, while linear and polynomial models are pulled by them.

### Why does Simple Linear Regression fail?

Vehicle depreciation is not linear in age. The first 3–5 years see the steepest price drop; after roughly 10 years the curve flattens. A straight line cannot represent this curvature.

### Why is Polynomial Regression an improvement but not a winner?

Adding quadratic terms captures the depreciation curve, raising R² substantially. But the polynomial model operates on numeric features only — it cannot represent the categorical structure of the data.

### Application to ride-hailing driver vehicle acquisition (Uganda)

This methodology has direct application to the Ugandan ride-hailing market. India's used-vehicle market shares structural features with Uganda's — right-hand-drive configuration, dominant Japanese manufacturers (Maruti-Suzuki, Toyota, Honda, Hyundai), import-driven price formation. A Random-Forest-based pricing model trained on locally scraped Jiji.ug data would enable Ugandan drivers to evaluate listings against an objective benchmark.

The next phase of this research will:
1. Replicate this methodology on a locally scraped dataset of 5,000+ Jiji.ug listings
2. Integrate the price model with ride-hailing earnings data from the TinkaTaxi platform
3. Build a public-facing tool that drivers can use before purchasing a vehicle

### Limitations

- Indian price levels are not directly transferable to Uganda even though market structure is similar
- Dataset reflects pre-2021 conditions; post-pandemic shifts are not captured
- Random Forest hyperparameters were not exhaustively tuned

### Final answer

**The best model for predicting used vehicle prices on this dataset is Random Forest Regression, decisively outperforming both Simple Linear Regression and Polynomial Regression. The same modeling approach is recommended for application to Uganda's used-vehicle market.**
```

### After defining the notebook, run it

```bash
jupyter nbconvert --to notebook --execute capstone_vehicle_prices.ipynb --output capstone_vehicle_prices.ipynb --ExecutePreprocessor.timeout=600
```

If any cell errors, FIX THE ERROR before continuing. Common issues:
- Missing column → check the actual columns in `Car details v3.csv` and adjust
- `sparse_output` vs `sparse` for OneHotEncoder → version-dependent; use `sparse_output=False` for sklearn >=1.2, `sparse=False` for older. Detect via `sklearn.__version__` and adapt.

Verify all expected output files exist:
```bash
ls -la vehicles_clean.csv rf_pipeline.joblib ui_metadata.json model_comparison.csv fig_*.png
```

---

## PHASE 4 — Build the Streamlit app

Create `app.py` at the project root. Use this exact content:

```python
"""
Used Vehicle Price Predictor — Streamlit UI
Capstone Project: Tinka Fahad & Tugume Andrew
Cavendish University Uganda, May 2026
"""

import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Used Vehicle Price Predictor — Capstone",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model_and_metadata():
    pipeline = joblib.load('rf_pipeline.joblib')
    with open('ui_metadata.json', 'r') as f:
        metadata = json.load(f)
    return pipeline, metadata

@st.cache_data
def load_cleaned_data():
    return pd.read_csv('vehicles_clean.csv')

try:
    pipeline, meta = load_model_and_metadata()
    df_clean = load_cleaned_data()
except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}. Run the notebook first.")
    st.stop()

INR_TO_UGX = 45
INR_TO_USD = 0.012

with st.sidebar:
    st.markdown("## 📚 About this project")
    st.markdown(
        """
        **Capstone Project**
        Predictive Analytics

        **Authors:**
        - Tinka Fahad (254430)
        - Tugume Andrew

        **Institution:**
        Cavendish University Uganda
        BSc Data Science & AI

        **Date:** May 2026
        """
    )
    st.markdown("---")
    st.markdown("## 🎯 Model performance")
    metrics = meta['model_metrics']
    st.metric("R² Score", f"{metrics['r2']:.3f}")
    st.metric("RMSE", f"₹{metrics['rmse_inr']:,.0f}")
    st.metric("MAE", f"₹{metrics['mae_inr']:,.0f}")
    st.markdown("---")
    st.markdown("## 📊 Dataset")
    st.markdown(
        f"""
        - **Source:** CarDekho India (Kaggle)
        - **Records:** {len(df_clean):,}
        - **Model:** Random Forest Regression
        """
    )

st.title("🚗 Used Vehicle Price Predictor")
st.markdown(
    """
    A Random Forest regression model predicting used vehicle prices,
    trained on the **CarDekho India** dataset. Designed for application to
    Uganda's used-vehicle market, where similar manufacturers
    (Maruti-Suzuki, Toyota, Honda, Hyundai) dominate the ride-hailing fleet.
    """
)
st.markdown("---")
st.markdown("## 🔧 Enter vehicle details")

col1, col2, col3 = st.columns(3)

with col1:
    manufacturer = st.selectbox(
        "Manufacturer", meta['manufacturers'],
        index=meta['manufacturers'].index('Toyota') if 'Toyota' in meta['manufacturers'] else 0
    )
    year = st.number_input(
        "Year of manufacture", min_value=meta['year_min'], max_value=meta['year_max'],
        value=min(2014, meta['year_max']), step=1
    )
    km_driven = st.number_input(
        "Kilometres driven", min_value=0, max_value=meta['km_driven_max'],
        value=80_000, step=1000
    )

with col2:
    fuel = st.selectbox("Fuel type", meta['fuel_types'])
    transmission = st.selectbox("Transmission", meta['transmissions'])
    owner = st.selectbox("Ownership history", meta['owner_types'])

with col3:
    engine_cc = st.number_input(
        "Engine size (CC)", min_value=meta['engine_cc_min'], max_value=meta['engine_cc_max'],
        value=1500, step=100
    )
    max_power = st.number_input(
        "Max power (bhp)", min_value=float(meta['max_power_min']), max_value=float(meta['max_power_max']),
        value=85.0, step=5.0
    )
    mileage = st.number_input(
        "Fuel efficiency (kmpl)", min_value=float(meta['mileage_min']), max_value=float(meta['mileage_max']),
        value=18.0, step=0.5
    )

col4, col5 = st.columns(2)
with col4:
    seats = st.selectbox(
        "Number of seats", meta['seats_options'],
        index=meta['seats_options'].index(5) if 5 in meta['seats_options'] else 0
    )
with col5:
    seller_type = st.selectbox("Seller type", meta['seller_types'])

st.markdown("---")

if st.button("📈 Predict price", type="primary", use_container_width=True):
    age = 2021 - year
    input_df = pd.DataFrame([{
        'age': age, 'km_driven': km_driven, 'mileage_kmpl': mileage,
        'engine_cc': engine_cc, 'max_power_bhp': max_power, 'seats': seats,
        'manufacturer': manufacturer, 'fuel': fuel, 'transmission': transmission,
        'seller_type': seller_type, 'owner': owner,
    }])

    predicted_price_inr = float(pipeline.predict(input_df)[0])
    predicted_price_ugx = predicted_price_inr * INR_TO_UGX
    predicted_price_usd = predicted_price_inr * INR_TO_USD

    st.markdown("## 💰 Predicted price")
    c1, c2, c3 = st.columns(3)
    c1.metric("In Indian Rupees", f"₹ {predicted_price_inr:,.0f}")
    c2.metric("In Ugandan Shillings (approx.)", f"UGX {predicted_price_ugx:,.0f}")
    c3.metric("In US Dollars (approx.)", f"$ {predicted_price_usd:,.0f}")

    st.caption(
        "_Currency conversions are approximate (May 2026 rates) for cross-market intuition. "
        "Actual Ugandan prices depend on import duties, local supply, and listing-specific "
        "factors not captured in this model._"
    )

    st.markdown("### 🔍 How this compares to similar vehicles in the dataset")
    similar = df_clean[
        (df_clean['manufacturer'] == manufacturer) &
        (df_clean['fuel'] == fuel) &
        (df_clean['age'].between(max(0, age - 2), age + 2))
    ]

    if len(similar) >= 5:
        avg_price = similar['selling_price'].mean()
        median_price = similar['selling_price'].median()
        pct_diff = ((predicted_price_inr - avg_price) / avg_price) * 100

        st.markdown(f"Found **{len(similar)} similar vehicles** ({manufacturer}, {fuel}, age ±2 years).")
        c1, c2 = st.columns(2)
        c1.metric("Average of similar", f"₹ {avg_price:,.0f}")
        c2.metric("Median of similar", f"₹ {median_price:,.0f}")

        if abs(pct_diff) < 10:
            st.success(f"✅ Predicted price within 10% of average for similar vehicles ({pct_diff:+.1f}%) — typical for this profile.")
        elif pct_diff > 0:
            st.info(f"📈 Predicted price {pct_diff:+.1f}% above average. May reflect higher power, lower km, or premium ownership history.")
        else:
            st.warning(f"📉 Predicted price {pct_diff:+.1f}% below average. May reflect high km, older ownership, or smaller engine.")
    else:
        st.info("Not enough similar vehicles for comparison. Prediction is based on broader model patterns.")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
        Predictive Analytics Capstone Project. Random Forest Regression trained on CarDekho India dataset.
        Methodology designed for transfer to Uganda's used-vehicle market.
    </div>
    """,
    unsafe_allow_html=True
)
```

Test that it launches without errors:

```bash
# Launch in background, wait 4 seconds, check it's running, kill it
streamlit run app.py --server.headless true &
STREAMLIT_PID=$!
sleep 4
if kill -0 $STREAMLIT_PID 2>/dev/null; then
    echo "✅ Streamlit launched successfully"
    kill $STREAMLIT_PID
else
    echo "❌ Streamlit failed to launch"
    exit 1
fi
```

---

## PHASE 5 — Generate the Word report

Create `generate_report.py`:

```python
import pandas as pd
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

results = pd.read_csv('model_comparison.csv')

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Calibri'
style.font.size = Pt(11)

# Title
title = doc.add_heading('Predicting Used Vehicle Prices in an Emerging Market', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_run = subtitle.add_run(
    'A Regression Analysis of Indian Used-Car Listings with Application to '
    "Uganda's Ride-Hailing Driver Vehicle Acquisition"
)
sub_run.italic = True
sub_run.font.size = Pt(13)

doc.add_paragraph()

author_para = doc.add_paragraph()
author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
author_para.add_run('Tinka Fahad — Student ID: 254430\n').bold = True
author_para.add_run('Tugume Andrew\n').bold = True
author_para.add_run('BSc Data Science & Artificial Intelligence\n')
author_para.add_run('Cavendish University Uganda\n')
author_para.add_run('May 2026')

doc.add_page_break()

doc.add_heading('Abstract', level=1)
doc.add_paragraph(
    "This study develops and compares three regression models — Simple Linear Regression, "
    "Polynomial Regression, and Random Forest Regression — to predict the resale price of used "
    "vehicles using the Indian CarDekho dataset (8,128 listings). Random Forest Regression achieved "
    f"R² = {results.iloc[2]['R²']:.3f} on held-out test data, with RMSE = "
    f"INR {results.iloc[2]['RMSE (INR)']:,.0f}, substantially outperforming the linear baseline "
    f"(R² = {results.iloc[0]['R²']:.3f}). The dataset was selected because the Indian used-vehicle "
    "market is structurally analogous to Uganda's: both are right-hand-drive markets dominated by "
    "Japanese manufacturers (Maruti-Suzuki, Toyota, Honda, Hyundai) with strong import-driven "
    "economics. Findings are directly applicable to Uganda's ride-hailing driver community."
)

doc.add_heading('1. Introduction and Motivation', level=1)
doc.add_paragraph(
    "As founder of TinkaTaxi, a Kampala-based ride-hailing platform serving over 3,000 users and "
    "2,000 drivers, and as an active driver on competing platforms, the lead author has observed "
    "first-hand that vehicle acquisition is the single largest capital decision a ride-hailing "
    "driver makes in the Ugandan market. Drivers routinely overpay for unreliable vehicles or "
    "undervalue serviceable ones because no transparent, data-driven pricing reference exists."
)
doc.add_paragraph(
    "Uganda's used-vehicle market is overwhelmingly composed of Japanese right-hand-drive imports. "
    "Direct Ugandan data of sufficient scale is not publicly available. India's used-car market "
    "is the closest structural analogue: right-hand-drive, dominated by Japanese manufacturers, "
    "and shaped by similar import-and-resale dynamics. The CarDekho dataset therefore provides "
    "the most relevant publicly available training data for a Ugandan price-prediction methodology."
)

doc.add_heading('2. Dataset and Data Preparation', level=1)
doc.add_paragraph(
    "The dataset is the 'Vehicle Dataset from CarDekho' published on Kaggle, specifically "
    "'Car details v3.csv' containing 8,128 used-vehicle listings from India's largest used-car "
    "marketplace, with 13 columns covering price, year, manufacturer, model, mileage, fuel type, "
    "transmission, ownership history, engine size, and engine power."
)
doc.add_paragraph(
    "Cleaning included: numeric extraction from mixed text columns (mileage 'kmpl', engine 'CC', "
    "max_power 'bhp'); manufacturer extraction from full vehicle name; removal of rows missing "
    "critical predictors; filtering of implausible prices (below INR 50,000 or above INR 10,000,000) "
    "and impossible odometer readings; vehicle year restricted to 1990–2021. Feature engineering "
    "derived vehicle age from year, log-transformed km_driven and selling_price."
)

doc.add_heading('3. Data Understanding and Exploration', level=1)
doc.add_paragraph(
    "Descriptive statistics revealed right-skew in price and km_driven, motivating log-transformations. "
    "The correlation heatmap showed strong negative correlation between price and age, strong positive "
    "correlations between price and engine power, and between price and engine size. Scatter plots "
    "revealed clearly non-linear relationships — directly motivating polynomial regression as Model 2."
)
doc.add_paragraph(
    "Top manufacturers in the dataset — Maruti, Hyundai, Honda, Toyota — overlap substantially with "
    "Uganda's used-vehicle fleet. This overlap is the empirical basis for treating the Indian market "
    "as a proxy for Uganda's."
)

doc.add_heading('4. Regression Models', level=1)

doc.add_heading('4.1 Model 1 — Simple Linear Regression', level=2)
doc.add_paragraph(
    "Simple Linear Regression using vehicle age as single predictor. Established a baseline. "
    f"Achieved R² = {results.iloc[0]['R²']:.3f} on held-out test set, with "
    f"RMSE = INR {results.iloc[0]['RMSE (INR)']:,.0f}. The poor fit reflects a fundamental shape "
    "mismatch between a straight line and the curved depreciation pattern in the data."
)

doc.add_heading('4.2 Model 2 — Polynomial Regression', level=2)
doc.add_paragraph(
    "Polynomial Regression of degree 2 using age, km_driven, max_power_bhp, and engine_cc as predictors, "
    "with feature scaling applied prior to polynomial expansion. Addresses both the curvature deficiency "
    f"and single-feature limitation of Model 1. Achieved R² = {results.iloc[1]['R²']:.3f} and "
    f"RMSE = INR {results.iloc[1]['RMSE (INR)']:,.0f} — substantial improvement over the linear baseline. "
    "Five-fold cross-validation confirmed stability."
)

doc.add_heading('4.3 Model 3 — Random Forest Regression', level=2)
doc.add_paragraph(
    "Random Forest with 200 estimators, max depth 20, minimum-leaf-size 3, using numeric features "
    "(age, km_driven, mileage, engine, max_power, seats) and categorical features (manufacturer, fuel, "
    f"transmission, seller_type, owner). Achieved R² = {results.iloc[2]['R²']:.3f} and "
    f"RMSE = INR {results.iloc[2]['RMSE (INR)']:,.0f} — best performance in this study. Feature "
    "importance analysis identified engine power, age, and manufacturer as strongest individual predictors."
)

doc.add_heading('5. Model Comparison', level=1)
doc.add_paragraph("Final performance of all three models on the held-out test set:")

table = doc.add_table(rows=1, cols=4)
table.style = 'Light Grid Accent 1'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Model'
hdr_cells[1].text = 'RMSE (INR)'
hdr_cells[2].text = 'MAE (INR)'
hdr_cells[3].text = 'R²'
for cell in hdr_cells:
    for para in cell.paragraphs:
        for run in para.runs:
            run.bold = True

for _, row in results.iterrows():
    row_cells = table.add_row().cells
    row_cells[0].text = str(row['Model'])
    row_cells[1].text = f"{row['RMSE (INR)']:,.0f}"
    row_cells[2].text = f"{row['MAE (INR)']:,.0f}"
    row_cells[3].text = f"{row['R²']:.3f}"

doc.add_heading('6. Conclusion', level=1)
doc.add_paragraph(
    "Random Forest Regression is the best-performing model for this problem and dataset. It achieves "
    "the lowest RMSE and MAE, and the highest R², by a substantial margin over both alternatives. "
    "Three factors explain its superiority: it natively handles categorical features that carry "
    "significant predictive weight (manufacturer, fuel, ownership history); it captures non-linear "
    "interaction effects between features; and it is robust to outliers."
)
doc.add_paragraph(
    "Simple Linear Regression fails because vehicle depreciation is fundamentally non-linear in age. "
    "Polynomial Regression captures curvature but cannot represent categorical structure, leaving "
    "substantial variance unexplained."
)
doc.add_paragraph(
    "This methodology has direct application to the Ugandan ride-hailing market. India's used-vehicle "
    "market shares structural features — right-hand-drive configuration, dominant Japanese manufacturers, "
    "import-driven price formation — with Uganda's. The next phase will replicate this methodology on "
    "locally scraped Jiji.ug listings and integrate the resulting price model with operational analytics "
    "from the TinkaTaxi platform."
)

doc.add_heading('7. Limitations and Future Work', level=1)
doc.add_paragraph(
    "Absolute price levels are India-specific. Dataset reflects pre-2021 conditions. Hyperparameter "
    "tuning was limited. Future work will extend the methodology to locally collected Ugandan data."
)

doc.add_heading('8. References', level=1)
doc.add_paragraph(
    "Birla, N. (2020). Vehicle Dataset from CarDekho. Kaggle. "
    "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"
)
doc.add_paragraph("Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.")
doc.add_paragraph("Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.")

doc.save('Capstone_Report.docx')
print("✅ Saved Capstone_Report.docx")
```

Run it:
```bash
python generate_report.py
ls -la Capstone_Report.docx
```

---

## PHASE 6 — Organize files

Move figures to a subfolder:

```bash
mkdir -p figures
mv fig_*.png figures/
ls figures/
```

---

## PHASE 7 — Create .gitignore, README, requirements.txt

`.gitignore`:
```
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
Car details v3.csv
car data.csv
CAR DETAILS FROM CAR DEKHO.csv
*.zip
.DS_Store
Thumbs.db
.vscode/
.idea/
```

`requirements.txt`:
```bash
pip freeze > requirements.txt
```

`README.md`:

```markdown
# Predicting Used Vehicle Prices in an Emerging Market

**A Regression Analysis of Indian Used-Car Listings with Application to Uganda's Ride-Hailing Driver Vehicle Acquisition**

> BSc Data Science & Artificial Intelligence — Predictive Analytics Capstone
> Cavendish University Uganda — May 2026

**Authors:**
- **Tinka Fahad** — Student ID: 254430
- **Tugume Andrew**

---

## Summary

This capstone develops three regression models — Simple Linear, Polynomial (degree 2), and Random Forest — to predict the resale price of used vehicles, using the CarDekho India dataset (~8,128 listings). Random Forest Regression achieves the best performance and is identified as the best model for the problem.

The methodology is framed for direct application to **Uganda's used-vehicle market**, which shares structural features with India's market: right-hand-drive configuration, dominant Japanese manufacturers (Maruti-Suzuki, Toyota, Honda, Hyundai), and strong import-driven price formation. A Streamlit web application demonstrates the trained model for end-users.

The lead author is founder of **TinkaTaxi**, a Kampala-based ride-hailing platform serving 3,000+ users and 2,000+ drivers, and an active driver on competing platforms. This domain experience grounds the work in real operational context.

---

## Marking Scheme Coverage

| Rubric Component | Marks | Location |
|---|---|---|
| Data Preparation & Preprocessing | 20 | Notebook Sections 2–4 |
| Data Understanding & Exploration | 20 | Notebook Sections 5–7 |
| Regression Model 1 (Simple Linear) | 20 | Notebook Section 8 |
| Regression Model 2 (Polynomial) | 20 | Notebook Section 9 |
| Regression Model 3 (Random Forest) + Conclusion | 20 | Notebook Sections 10–12 |
| **Total** | **100** | |

---

## Repository structure

- `capstone_vehicle_prices.ipynb` — Main analysis notebook (all 12 sections)
- `app.py` — Streamlit web application
- `generate_report.py` — Generates the Word report
- `Capstone_Report.docx` — Written submission report
- `vehicles_clean.csv` — Cleaned dataset (output of notebook)
- `model_comparison.csv` — Final metrics for all three models
- `rf_pipeline.joblib` — Trained Random Forest pipeline
- `ui_metadata.json` — UI dropdown options
- `requirements.txt` — Python dependencies (pinned)
- `figures/` — All generated visualisations

The raw dataset (`Car details v3.csv`) is not included; instructions to download it below.

---

## How to reproduce

### Prerequisites
- Python 3.9 or newer
- A free Kaggle account

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/used-vehicle-price-prediction-capstone.git
cd used-vehicle-price-prediction-capstone
```

### 2. Download the dataset
From https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho — download and extract so that `Car details v3.csv` is in the project root.

### 3. Set up Python environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook capstone_vehicle_prices.ipynb
# Run all cells (Cell → Run All)
```

### 5. Launch the web app
```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 6. (Optional) Regenerate the Word report
```bash
python generate_report.py
```

---

## Key Findings

1. **Random Forest Regression is the best model**, decisively outperforming both the Simple Linear baseline and the Polynomial model.
2. **Categorical features matter** — manufacturer carries substantial predictive weight that linear models cannot capture well.
3. **Depreciation is non-linear in age** — the first 3–5 years see the steepest price drops; the curve then flattens.
4. **The Indian market is a valid proxy for the Ugandan ride-hailing market** — same dominant manufacturers, same right-hand-drive configuration, similar import-driven economics.

---

## Future Work

1. Replicate this methodology on a locally scraped Jiji.ug dataset
2. Integrate the price model with operational analytics from the TinkaTaxi ride-hailing platform
3. Deploy a public-facing tool for Ugandan ride-hailing drivers

---

## Citation

**Dataset:** Birla, N. (2020). *Vehicle Dataset from CarDekho.* Kaggle. https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

**Lead author's industry context:** TinkaTaxi (https://tinkataxi.com), a Kampala-based ride-hailing platform.

---

## Contact

- **Tinka Fahad** — Student ID 254430 — Cavendish University Uganda
- **Tugume Andrew** — Cavendish University Uganda

Submission: May 2026
```

---

## PHASE 8 — Git init and GitHub push

```bash
git init
git add .
git status   # verify Car details v3.csv is NOT listed
git commit -m "Initial capstone submission — Tinka Fahad & Tugume Andrew"
```

Then ask the user (only now):

> "I need two things from you:
> 1. **GitHub username** — what's your GitHub handle?
> 2. **A Personal Access Token** — go to https://github.com/settings/tokens, click 'Generate new token (classic)', give it 'repo' scope, copy the token. Paste it when I prompt.
>
> Once you give me your username, I'll create the repo and push everything."

After getting the username, use the GitHub CLI (`gh`) if available, otherwise use curl:

```bash
# Try gh first (cleanest)
if command -v gh &> /dev/null; then
    gh auth status || gh auth login
    gh repo create used-vehicle-price-prediction-capstone --public --source=. --remote=origin --push --description "BSc Data Science Capstone — Predicting used vehicle prices. Tinka Fahad & Tugume Andrew, Cavendish University Uganda."
else
    # Fallback to manual: just set up the remote and prompt for token
    echo "TINKA: Please paste your GitHub Personal Access Token here. It will not be saved."
    read -s GH_TOKEN

    # Create repo via API
    curl -u "$USERNAME:$GH_TOKEN" https://api.github.com/user/repos \
        -d '{"name":"used-vehicle-price-prediction-capstone","description":"BSc Data Science Capstone — Tinka Fahad & Tugume Andrew","private":false}'

    git remote add origin "https://$USERNAME:$GH_TOKEN@github.com/$USERNAME/used-vehicle-price-prediction-capstone.git"
    git branch -M main
    git push -u origin main
fi
```

After push completes, confirm with:

```bash
echo ""
echo "✅ Repository pushed: https://github.com/$USERNAME/used-vehicle-price-prediction-capstone"
echo ""
```

---

## PHASE 9 — Final summary

Print this exactly:

```
==========================================================
✅ CAPSTONE BUILD COMPLETE
==========================================================

Project folder: ~/capstone-vehicle-prices
GitHub repo:    https://github.com/$USERNAME/used-vehicle-price-prediction-capstone

Files created:
  ✅ capstone_vehicle_prices.ipynb (notebook with 3 regression models)
  ✅ app.py (Streamlit web application)
  ✅ Capstone_Report.docx (written report)
  ✅ rf_pipeline.joblib (saved trained model)
  ✅ vehicles_clean.csv (cleaned dataset)
  ✅ model_comparison.csv (final metrics)
  ✅ figures/ (11 PNG visualisations)
  ✅ README.md (lecturer-facing reproducibility guide)
  ✅ requirements.txt
  ✅ .gitignore

==========================================================
HOW TO DEMO (for your video):
==========================================================

1. Open GitHub repo in browser — show the README
2. Open the notebook — scroll through it
3. Launch the web app:
     cd ~/capstone-vehicle-prices
     source venv/bin/activate
     streamlit run app.py
   Then go to http://localhost:8501 in your browser

4. Demo inputs (Kampala ride-hailing vehicle profile):
   - Manufacturer: Toyota
   - Year: 2014
   - Km driven: 80000
   - Fuel: Petrol
   - Transmission: Manual
   - Engine: 1500
   - Max power: 85
   - Mileage: 18
   - Seats: 5
   - Owner: First Owner
   - Seller type: Individual

5. Click 'Predict price' and discuss the result with the lecturer

==========================================================
TO SUBMIT:
==========================================================

Email the lecturer:
  - Attach: Capstone_Report.docx
  - Include: GitHub repo URL
  - Include: Video walkthrough URL (YouTube unlisted or Drive)

Tinka — everything is ready. Good luck.
```

---

## Notes for you (Claude Code) on error handling

- **If a notebook cell errors:** read the error, fix the cell, re-run. Common issues: sklearn version (sparse_output vs sparse), pandas version (deprecation warnings), missing columns.
- **If Streamlit errors on launch:** check that `rf_pipeline.joblib` and `ui_metadata.json` exist. If not, the notebook didn't finish — go back and re-run.
- **If git push fails:** likely auth. Use a Personal Access Token, not a password.
- **If GitHub repo creation fails:** the repo name might already exist on the user's account. Append a number: `used-vehicle-price-prediction-capstone-2`.
- **NEVER stop halfway and leave the user with a half-finished project.** If you cannot complete a phase, complete every other phase first, then come back and clearly explain what's blocking you and what the user needs to do.

Good luck. Build this all the way through.
