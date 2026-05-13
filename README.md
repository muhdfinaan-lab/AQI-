# 🌫️ Air Quality Index (AQI) Prediction using Machine Learning

> A machine learning project to classify and predict Air Quality Index (AQI) categories from real-time pollutant data collected across various locations in India.

---

## 📌 Project Overview

Air pollution is one of the most critical environmental challenges in India. This project analyzes **real-time air quality data** from monitoring stations across multiple Indian states and cities. Using various machine learning classification algorithms, the model predicts the **AQI category** of a location based on pollutant readings — helping citizens and authorities take timely action.

---

## 📂 Dataset

- **File:** `REAL TIME AIR QUALITY FROM VARIOUS LOCATION IN INDIA.csv`
- **Total Records:** 3,322 rows × 11 columns
- **Source:** Real-time air quality monitoring stations across India
- **Last Updated:** February 2026

### 📋 Features

| Column | Description |
|---|---|
| `country` | Country name (India) |
| `state` | State of the monitoring station |
| `city` | City of the monitoring station |
| `station` | Exact station name |
| `last_update` | Timestamp of the last reading |
| `latitude` | Geographic latitude of the station |
| `longitude` | Geographic longitude of the station |
| `pollutant_id` | Type of pollutant (NO2, SO2, CO, PM10, PM2.5, NH3, OZONE) |
| `pollutant_min` | Minimum pollutant concentration recorded |
| `pollutant_max` | Maximum pollutant concentration recorded |
| `pollutant_avg` | Average pollutant concentration recorded |

### 🗺️ States Covered
Bihar, Assam, Andhra Pradesh, Andaman & Nicobar, Delhi, Chhattisgarh, Chandigarh, Gujarat, Haryana, Karnataka, Kerala, Madhya Pradesh, Himachal Pradesh, Jharkhand, Maharashtra, Odisha, Puducherry, Punjab, Meghalaya, West Bengal, and more.

### 🧪 Pollutants Monitored
`NO2` · `SO2` · `CO` · `PM10` · `PM2.5` · `NH3` · `OZONE`

---

## 🎯 Target Variable — AQI Categories

| Label | Category | Description |
|---|---|---|
| 0 | Good | Air quality is satisfactory; little to no risk |
| 1 | Satisfactory | Acceptable air quality |
| 2 | Moderate | Some pollutants may pose moderate health concerns |
| 3 | Poor | Health effects may be experienced |
| 4 | Very Poor | Health alert; serious risk for everyone |

---

## 🔧 Tech Stack

- **Language:** Python 3.13
- **Environment:** Jupyter Notebook (Conda)
- **Libraries:**
  - `pandas`, `numpy` — Data manipulation
  - `matplotlib`, `seaborn` — Data visualization
  - `scikit-learn` — Machine learning models & preprocessing
  - `xgboost` — Gradient boosting
  - `joblib` — Model serialization

---

## 🔄 Project Workflow

```
Raw Data → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
```

### 1. 📊 Exploratory Data Analysis (EDA)
- Dataset shape, data types, and null value inspection
- Statistical summary (`describe()`)
- Distribution analysis of pollutants across states and cities
- Boxplots before and after outlier treatment
- Correlation heatmaps
- Pollutant-wise and state-wise visualizations using Seaborn and Matplotlib

### 2. 🧹 Data Preprocessing
- **Missing Value Treatment:** Handled null values in `pollutant_min`, `pollutant_max`, and `pollutant_avg`
- **Outlier Removal:** IQR-based outlier detection and capping
- **Encoding:**
  - `OneHotEncoder` applied to `state`, `city`, and `pollutant_id`
- **Scaling:**
  - `StandardScaler` applied to numerical features (`latitude`, `longitude`, `pollutant_min`, `pollutant_max`, `pollutant_avg`)
- **Class Imbalance Handling:** Resampling techniques (SMOTE / oversampling) applied to balance AQI categories

### 3. 🤖 Models Trained

| Model | Test Accuracy |
|---|---|
| Logistic Regression | — |
| Decision Tree | — |
| K-Nearest Neighbors (KNN) | — |
| Naive Bayes (GaussianNB) | ~87.1% |
| Support Vector Machine (SVM - Linear) | — |
| Support Vector Machine (SVM - RBF, Tuned) | ~90.0% |
| Random Forest | — |
| AdaBoost | ~84.9% |
| Gradient Boosting | ~93.5% ✅ |
| XGBoost | ~93.2% |

> ✅ **Best Model: Gradient Boosting Classifier** with ~93.5% test accuracy

### 4. ⚙️ Hyperparameter Tuning
- **GridSearchCV** applied to SVM (RBF kernel) for optimal `C` and `gamma`
- Best SVM estimator retrained and evaluated

### 5. 💾 Model Saving
Trained models and encoders saved using `joblib`:

```
best_model.pkl
scaler.pkl
ohe_state.pkl
ohe_city.pkl
ohe_pollutant_id.pkl
train_columns.pkl
```

---

## 🖥️ Prediction Interface

A custom `run_model()` function allows interactive AQI prediction from user input:

```
Enter AQI Prediction Details
Enter State       : Kerala
Enter City        : Kochi
Enter Latitude    : 9.93
Enter Longitude   : 76.26
Enter Pollutant   : NH3
Enter Pollutant Min : 20.0
Enter Pollutant Max : 80.0
Enter Pollutant Avg : 45.2

========================================
         AQI PREDICTION RESULT
========================================
State         : Kerala
City          : Kochi
Pollutant     : NH3
Pollutant Min : 20.0
Pollutant Max : 80.0
Pollutant Avg : 45.2
========================================
Predicted AQI : Good
Message       : Air Quality is Good!
========================================
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
```

### Run the Notebook
```bash
jupyter notebook AQI_PROJECT_.ipynb
```

### Load the Model for Prediction
```python
import joblib

model           = joblib.load('best_model.pkl')
scaler          = joblib.load('scaler.pkl')
ohe_state       = joblib.load('ohe_state.pkl')
ohe_city        = joblib.load('ohe_city.pkl')
ohe_pollutant   = joblib.load('ohe_pollutant_id.pkl')
train_columns   = joblib.load('train_columns.pkl')
```

---

## 📈 Results Summary

- **Best Model:** Gradient Boosting Classifier
- **Test Accuracy:** ~93.5%
- **AQI Categories Predicted:** Good, Satisfactory, Moderate, Poor, Very Poor
- **Evaluation Metrics:** Accuracy Score, Confusion Matrix, Classification Report (Precision, Recall, F1-Score)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

> ⭐ If you found this project helpful, please give it a star!
