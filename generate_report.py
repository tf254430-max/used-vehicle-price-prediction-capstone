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
    "vehicles — using the Indian CarDekho dataset (8,128 listings). Random Forest Regression achieved "
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
