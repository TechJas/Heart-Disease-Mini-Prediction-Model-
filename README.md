# Heart Disease Prediction Model

**Author:** Jasmin Banu M

A machine learning model that predicts heart disease from clinical patient data using Random Forest, deployed as an interactive Streamlit web app.

---

## The Problem

Heart disease is the leading cause of death globally. Early detection saves lives, but manual diagnosis requires specialists and time. This project builds an ML model that predicts heart disease from 8 clinical features — enabling faster screening.

## What This Project Does

- **Analyzes** a medical dataset of 1,319 patients with 8 clinical features
- **Handles outliers** using IQR method with domain-aware decisions (retains clinically meaningful values)
- **Tunes** a Random Forest classifier using GridSearchCV with 5-fold stratified cross-validation
- **Achieves 98.47% F1 score** on held-out test data
- **Deploys** as a Streamlit web app for real-time prediction

## Results

| Metric | Score |
|--------|-------|
| CV F1 Score | 99.07% |
| Test F1 Score | 98.47% |
| Precision | 98% |
| Recall | 98% |
| Total Misclassifications | 5 out of 264 |

### Feature Importance

| Feature | Importance |
|---------|-----------|
| Troponin | ~57% |
| CK-MB | ~26% |
| Age | ~5% |
| Others | ~12% |

Troponin and CK-MB are cardiac biomarkers that naturally elevate during heart muscle damage — the model learned clinically meaningful patterns, not data artifacts.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python |
| Model | Random Forest (tuned) |
| Preprocessing | StandardScaler, LabelEncoder |
| Evaluation | GridSearchCV, StratifiedKFold |
| Deployment | Streamlit |
| ML Library | Scikit-learn |

## Project Structure

```
Heart-Disease-Mini-Prediction-Model-/
├── app.py                    # Streamlit web app
├── heart_model.pkl           # Trained Random Forest model
├── scaler.pkl                # StandardScaler
├── requirements.txt          # Dependencies
├── heart_cleaned.csv         # Cleaned dataset
├── heart_dataset_expo.ipynb  # Full analysis notebook
├── Medicaldataset.csv        # Original dataset
└── README.md
```

## Installation

```bash
git clone https://github.com/TechJas/Heart-Disease-Mini-Prediction-Model-.git
cd Heart-Disease-Mini-Prediction-Model-
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

Opens at http://localhost:8501 — enter patient details in the sidebar and click Predict.

## Live Demo

Deployed on Streamlit Cloud: [https://heart-disease-mini-prediction-model.streamlit.app](https://heart-disease-mini-prediction-model.streamlit.app)

## Features Used

| Feature | Description |
|---------|------------|
| age | Patient age (years) |
| gender | 1 = Male, 0 = Female |
| heart_rate | Heart rate (bpm) |
| systolic_blood_pressure | Systolic BP (mmHg) |
| diastolic_blood_pressure | Diastolic BP (mmHg) |
| blood_sugar | Blood sugar (mg/dL) |
| ck_mb | CK-MB enzyme level (ng/mL) |
| troponin | Troponin enzyme level (ng/mL) |

## Key Decisions Made During Development

- **Outliers retained:** Extreme values in troponin, CK-MB, and blood pressure were kept because they represent clinically meaningful observations (e.g., elevated troponin indicates cardiac damage)
- **No data leakage:** Verified that biomarker separation between classes is legitimate clinical signal, not target leakage
- **Scaling kept:** StandardScaler included for KNN comparison fairness, even though Random Forest doesn't require it

## License

MIT License

## Disclaimer

This model is not meant to replace medical diagnosis. It assists in early detection through data-driven insights. Always consult healthcare professionals for medical decisions.
