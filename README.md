🫀 Heart Disease Prediction Model
👩‍💻 Author: Jasmin Banu M

A predictive machine learning model built using Python, Scikit-learn, and Random Forest Classifier to detect heart disease based on patient medical data.
Works seamlessly on both Google Colab and VS Code environments.

🚀 Features

🧠 Machine Learning Powered — Uses a Random Forest Classifier for accurate heart disease prediction.

📂 Smart Dataset Handling — Supports both CSV and JSON uploads and auto-saves data for reuse.

⚙️ Cross-Platform Compatibility — Runs smoothly on Google Colab or local Python environments.

🔐 Data Persistence — Saves preprocessed datasets and trained models using joblib.

📊 Interactive Prediction Interface — Prompts user input for real-time predictions.

💾 Automatic Scaling — Normalizes feature values using StandardScaler.

🧩 Project Workflow

Upload or Load Dataset

Upload a CSV/JSON medical dataset.

The program will save it as medical_dataset_saved.pkl for future use.

Data Preprocessing

Cleans column names.

Encodes the result column (positive = 1, negative = 0).

Scales numerical features for model stability.

Model Training

Splits data (80% training, 20% testing).

Trains using RandomForestClassifier.

Displays training accuracy.

Prediction Phase

Takes new patient details (age, gender, blood pressure, etc.).

Predicts disease status and confidence level.

🧰 Tech Stack
Component	Technology Used
Language	Python
Libraries	Pandas, NumPy, Scikit-learn, Joblib
Model	RandomForestClassifier
Environment	Google Colab / VS Code
📦 Installation
# Clone the repository
git clone https://github.com/<your-username>/heart-disease-prediction-model.git

# Navigate into project folder
cd heart-disease-prediction-model

# Install dependencies
pip install -r requirements.txt

🧾 Usage
▶️ Run the model
python heart_disease_predictor.py


You’ll be prompted to:

Upload a dataset (CSV/JSON)

Train the model automatically

Enter patient data for prediction

🧮 Example Features (Expected Columns)
Feature	Description
age	Patient’s age (in years)
gender	1 = Male, 0 = Female
heart_rate	Beats per minute
systolic_blood_pressure	Systolic BP (mmHg)
diastolic_blood_pressure	Diastolic BP (mmHg)
blood_sugar	Blood sugar level (mg/dL)
ck-mb	CK-MB enzyme level (ng/mL)
troponin	Troponin enzyme level (ng/mL)
result	Target label: 1 = Disease, 0 = No Disease
🧠 Output Example
✅ Model trained successfully! Accuracy: 94.50%

🩺 Enter new patient details for prediction:
Age (years): 45
Gender (1=Male, 0=Female): 1
Heart rate (bpm): 90
Systolic blood pressure (mmHg): 140
Diastolic blood pressure (mmHg): 90
Blood sugar (mg/dL): 180
CK-MB (ng/mL): 25
Troponin (ng/mL): 0.05

🔍 Prediction Result: 🩸 Positive (Heart Disease Detected)
📊 Confidence: 88.23%

📘 requirements.txt

Create a requirements.txt file containing:

pandas
numpy
scikit-learn
joblib
tk


💡 Note: tk is optional — used only for file dialog boxes in VS Code.

🧑‍🔬 Future Enhancements

✅ Integrate with a web interface (Flask/Streamlit) for user-friendly access.

🧠 Add explainable AI (XAI) visualizations using SHAP or LIME.

💻 Deploy as a cloud API for hospital integration.

📱 Create a mobile app version using React Native or Flutter.

🪪 License

This project is licensed under the MIT License — feel free to modify, use, and distribute with credit.

💬 Author Note

“This model is not meant to replace medical diagnosis but to assist in early detection through data-driven insights. Always consult healthcare professionals for medical decisions.”

— Jasmine Banu M

🧷 Tags

#AIForHealth #HeartDiseasePrediction #MachineLearning #DataScience #HealthcareAI

Would you like me to generate all the files (README.md + requirements.txt) as downloadable .zip so you can upload directly to GitHub?

Voice chat ended
