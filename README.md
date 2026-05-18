# Predicting Used Vehicle Prices in an Emerging Market

**A Regression Analysis of Indian Used-Car Listings with Application to Uganda's Ride-Hailing Driver Vehicle Acquisition**

> BSc Data Science & Artificial Intelligence — Predictive Analytics Capstone
> Cavendish University Uganda — May 2026

**Authors:**
- **Tinka Fahad** — Student ID: 254430
- **Tugume Andrew**

---

## Summary

This capstone develops three regression models — Simple Linear, Polynomial (degree 2), and Random Forest — to predict the resale price of used vehicles, using the CarDekho India dataset (8,128 listings; 7,857 after cleaning). Random Forest Regression achieves the best performance and is identified as the best model for the problem.

The methodology is framed for direct application to **Uganda's used-vehicle market**, which shares structural features with India's market: right-hand-drive configuration, dominant Japanese manufacturers (Maruti-Suzuki, Toyota, Honda, Hyundai), and strong import-driven price formation. A Streamlit web application demonstrates the trained model for end-users.

## Headline results (held-out test set)

| Model | R² | RMSE (INR) | MAE (INR) |
|---|---|---|---|
| Simple Linear (age only) | 0.175 | 723,055 | 382,268 |
| Polynomial degree 2 (4 features) | 0.847 | 311,945 | 181,319 |
| **Random Forest (11 features)** | **0.969** | **140,173** | **72,069** |

Random Forest wins by a large margin: it handles categorical features (manufacturer, fuel, ownership) natively, captures non-linear feature interactions, and is robust to outliers.

## Marking scheme coverage (Total: 100)

| Rubric Component | Marks | Where to find it |
|---|---|---|
| Data Preparation & Preprocessing | 20 | Notebook §2–4, Report §2 |
| Data Understanding & Exploration | 20 | Notebook §5–7, Report §3 |
| Regression Model 1 — Simple Linear | 20 | Notebook §8, Report §4.1 |
| Regression Model 2 — Polynomial | 20 | Notebook §9, Report §4.2 |
| Regression Model 3 — Random Forest + Conclusion | 20 | Notebook §10–12, Report §4.3–6 |

---

## Repository structure

- `capstone_vehicle_prices.ipynb` — Main analysis notebook (all 12 sections)
- `app.py` — Streamlit web application
- `generate_report.py` — Generates the Word report locally (not committed to this repository)
- `vehicles_clean.csv` — Cleaned dataset (output of notebook)
- `model_comparison.csv` — Final metrics for all three models
- `rf_pipeline.joblib` — Trained Random Forest pipeline
- `ui_metadata.json` — UI dropdown options
- `requirements.txt` — Python dependencies (minimum versions tested)
- `figures/` — All generated visualisations

## How to reproduce

### Prerequisites
- Python 3.9 or newer
- A free Kaggle account

### 1. Set up the Python environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Download the dataset
From https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho — download and extract so that `Car details v3.csv` is in the project root.

### 3. Build and execute the notebook
```powershell
python build_notebook.py
jupyter nbconvert --to notebook --execute capstone_vehicle_prices.ipynb --output capstone_vehicle_prices.ipynb --ExecutePreprocessor.timeout=600
```

### 4. Launch the web app
```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
# Open http://localhost:8501
```

### 5. Regenerate the Word report
```powershell
python generate_report.py
```

---

## Local usage
Use the project locally with these commands once the environment is installed and the dataset is available.

1. Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```
2. Execute the notebook end-to-end:
```powershell
python build_notebook.py
jupyter nbconvert --to notebook --execute capstone_vehicle_prices.ipynb --output capstone_vehicle_prices.ipynb --ExecutePreprocessor.timeout=600
```
3. Start the Streamlit app:
```powershell
streamlit run app.py
```
4. Open the app at `http://localhost:8501`.

---

## Contact
- **Tinka Fahad** — Student ID 254430 — Cavendish University Uganda
- **Tugume Andrew** — Cavendish University Uganda
