# How to Run the App on Your Laptop

A step-by-step guide for running the **Used Vehicle Price Predictor** Streamlit app.

> **Good to know:** the trained model ships inside this repository, so you do **not**
> need the dataset or to run the notebook. You only need Python + the libraries, then
> launch the app. Works on Windows, Mac, and Linux.

**Repository:** https://github.com/tf254430-max/used-vehicle-price-prediction-capstone

---

## Step 1 — Install Python (skip if already installed)
- Download Python 3.9 or newer from https://www.python.org/downloads/
- **Windows:** on the first install screen, tick **"Add Python to PATH"** before clicking Install. This one checkbox prevents most problems.

## Step 2 — Get the project files
**Easiest way (no Git needed):**
1. Open the repository link above.
2. Click the green **"<> Code"** button → **"Download ZIP"**.
3. Right-click the ZIP → **Extract All** → choose a simple location (e.g. Desktop).

**Or, with Git installed:**
```
git clone https://github.com/tf254430-max/used-vehicle-price-prediction-capstone.git
```

## Step 3 — Open a terminal inside the project folder
- Open the extracted folder (`used-vehicle-price-prediction-capstone` or `...-main`).
- **Windows:** click in the folder's address bar, type `cmd`, press **Enter**.
- **Mac/Linux:** open Terminal and `cd` into the folder.

## Step 4 — Install the required libraries (1–2 minutes)
```
pip install -r requirements.txt
```
If `pip` isn't recognised, use `python -m pip install -r requirements.txt`.

## Step 5 — Run the app
```
streamlit run app.py
```
It opens automatically at **http://localhost:8501**. If not, paste that address into any browser.
If `streamlit` isn't recognised, use `python -m streamlit run app.py`.

## To stop the app
Return to the terminal and press **Ctrl + C**.

---

## Once it's open
- **Tab 1 — Predict Price:** enter vehicle details → click **Predict price** → get a price in UGX plus a comparison against similar real listings.
- **Tab 2 — Explore the Data:** descriptive statistics and exploratory charts.
- **Tab 3 — Model Comparison:** the three-model results table and figures showing why Random Forest (R² = 0.969) wins.

## Troubleshooting
- **`python`/`pip` not recognised:** Python wasn't added to PATH in Step 1 — reinstall and tick that box, or use the `python -m ...` forms above.
- You do **not** need the dataset or the notebook — the trained model is already in the repo.
- Internet is only needed for Step 4. After that the app runs offline.
