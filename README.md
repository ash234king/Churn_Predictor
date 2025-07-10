# 🔍 Churn Predictor App

A powerful and easy-to-use **Streamlit web app** that predicts customer churn using a trained **Artificial Neural Network (ANN)** model. The app uses features like geography, age, balance, and more to predict whether a customer is likely to leave a business.

![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-blue)
![Keras](https://img.shields.io/badge/Backend-Keras-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 🚀 Live Demo

👉 [Live App on Streamlit Cloud]https://churnpredictor-mff4h7sfttwq4ephhgmgak.streamlit.app/ 


---

## 🧠 About the Model

The model is trained on a **real-world churn dataset** using an ANN architecture built with Keras. Key features include:

- Credit Score
- Geography (One-hot encoded)
- Gender (Label encoded)
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

Prediction Output:
- `1` = Will churn  
- `0` = Will stay

---

## 🛠 Features

- ✅ Clean and interactive Streamlit UI
- ✅ Real-time churn prediction
- ✅ EDA script for visual analysis
- ✅ Reproducible training pipeline
- ✅ Docker support
- ✅ GitHub Actions workflow for CI/CD

---

## 📁 Project Structure
churn-predictor-app/
│
├── app.py                    # Entry point: Streamlit frontend
├── prediction.py             # Logic to load model and make predictions
├── eda.py                    # EDA visualizations and summaries
├── experiments.py            # Model training script
│
├── data/
│   └── Churn_Modelling.csv   # Raw dataset
│
├── models/
│   ├── model.keras               # Trained ANN model
│   ├── scaler.pkl                # StandardScaler
│   ├── onehot_encoder_geo.pkl    # Geography encoder
│   ├── label_encoder_gender.pkl  # Gender encoder
│   └── feature_columns.pkl       # Feature column names
│
├── .github/
│   └── workflows/
│       └── streamlit-deploy.yml  # CI/CD workflow
│
├── Dockerfile                # Docker setup
├── requirements.txt          # Dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # Project README

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/churn-predictor-app.git
cd churn-predictor-app
pip install -r requirements.txt