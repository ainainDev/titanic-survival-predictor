# 🚢 Titanic Survival Prediction

A complete, leakage-safe machine learning pipeline that predicts whether a Titanic passenger would have survived, based on features like class, sex, age, and fare — built with **Scikit-learn** and deployed as an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project walks through an end-to-end ML workflow on the classic Titanic dataset — from raw data to a deployable prediction app — with a strong focus on **avoiding data leakage** and building a **production-ready pipeline** rather than a one-off notebook script.

**Live demo:** *https://aindev-titanic-survival-predictor.streamlit.app/*

---

## ✨ Features

- 🧹 Clean feature selection (drops leakage-prone and redundant columns)
- 🔧 Fully unified `sklearn.Pipeline` — imputation, encoding, scaling, and power transformation all bundled with the model
- 🚫 Zero data leakage — every preprocessing step is fit **only** on training data
- 📊 Model evaluated via accuracy, classification report, confusion matrix, and 5-fold cross-validation
- 🖥️ Interactive Streamlit interface for real-time predictions on custom passenger inputs

---

## 🗂️ Project Structure

```
titanic-survival-predictor/
├── app.py                                  # Streamlit web app
├── full_pipeline.pkl                       # Trained, serialized sklearn pipeline
├── Titanic_Survival_Prediction_Model.ipynb # Full training notebook (EDA → model)
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Pipeline

The trained pipeline (`full_pipeline`) chains every preprocessing step with the model, so raw input goes in and a prediction comes out — no manual preprocessing required at inference time.

```
Raw Passenger Data
        │
        ▼
Missing Value Imputation      (Age → median, Embarked → most frequent)
        │
        ▼
One-Hot Encoding              (Sex, Embarked)
        │
        ▼
Min-Max Scaling
        │
        ▼
Yeo-Johnson Power Transform   (Age, SibSp, Parch, Fare — fixes right-skew)
        │
        ▼
Logistic Regression
        │
        ▼
Survival Prediction
```

### Features Used
| Feature | Description |
|---|---|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | Passenger sex |
| `age` | Age in years |
| `sibsp` | # of siblings/spouses aboard |
| `parch` | # of parents/children aboard |
| `fare` | Ticket fare |
| `embarked` | Port of embarkation (S, C, Q) |

**Target:** `survived` (0 = No, 1 = Yes)

> Columns like `alive` (direct target leakage), `who`, `class`, `embark_town` (redundant), and `deck` (mostly missing) were deliberately excluded.

---

## 📈 Model Performance

Evaluated on a held-out test set (20% split, never used during training or preprocessing fitting):

| Metric | Score |
|---|---|
| **Test Accuracy** | ~77% |
| **5-Fold CV Mean Accuracy** | ~77% |

```
              precision    recall  f1-score   support

           0       0.78      0.83      0.80        88
           1       0.76      0.70      0.73        69

    accuracy                           0.77       157
```

> A simple Logistic Regression baseline — not tuned for maximum performance, but built with correct methodology (no leakage, proper evaluation) as the primary goal.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/titanic-survival-predictor.git
cd titanic-survival-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Using the App

1. Enter passenger details — class, sex, age, family aboard, fare, and port of embarkation.
2. Click **Predict Survival**.
3. View the prediction along with the model's confidence for each outcome.

---

## 🔬 Re-training the Model

To retrain from scratch or experiment with the pipeline:
1. Open `Titanic_Survival_Prediction_Model.ipynb` in Jupyter.
2. Run all cells — this loads the Titanic dataset (via Seaborn), builds and fits `full_pipeline`, and evaluates it.
3. Save the updated model:
   ```python
   import pickle
   with open('full_pipeline.pkl', 'wb') as f:
       pickle.dump(full_pipeline, f)
   ```
4. Restart the Streamlit app to use the newly trained model.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — preprocessing pipeline & Logistic Regression model
- **Seaborn / Matplotlib** — EDA and visualization
- **Streamlit** — web app interface

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning via `GridSearchCV`
- [ ] Try additional models (Random Forest, XGBoost) and compare performance
- [ ] Add feature importance visualization to the app
- [ ] Deploy permanently via Streamlit Community Cloud
- [ ] Add unit tests for the preprocessing pipeline

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and share.

---

## 🙌 Acknowledgements

- Dataset: [Seaborn's built-in Titanic dataset](https://github.com/mwaskom/seaborn-data) (derived from the original Kaggle Titanic competition data)
- Built as a learning project to practice leakage-safe ML pipeline design with Scikit-learn
