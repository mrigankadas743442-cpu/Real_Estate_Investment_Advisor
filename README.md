# 🏡 Real Estate Investment Advisor
## Predicting Property Investment Potential & Future Value

---

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Computing-013243?style=flat-square&logo=numpy&logoColor=white) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) ![MLflow](https://img.shields.io/badge/MLflow-Model%20Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-Model%20Persistence-3776AB?style=flat-square) ![Skops](https://img.shields.io/badge/Skops-Model%20Serialization-4B8BBE?style=flat-square) ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square&logo=python&logoColor=white) ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Charts-4C72B0?style=flat-square&logo=python&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-Scientific%20Computing-8CAAE6?style=flat-square&logo=scipy&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white) ![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?style=flat-square&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white) ![Streamlit Cloud](https://img.shields.io/badge/Streamlit%20Cloud-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

A machine learning-powered real estate analytics platform that analyzes property investment potential and estimates future property values. It uses a cleaned dataset containing **250,000 records and 43 columns**, applies classification and regression techniques, and presents interactive insights through a **Streamlit dashboard**.

---

## 🚀 Live Demo

👉 **[Try Real Estate Investment Advisor](https://realestateinvestmentadvisor743442piku.streamlit.app/)**

Explore property investment predictions, market insights, estimated future property values, and machine learning model performance through an interactive dashboard.

---

## 🎯 Project Objectives

- Analyze real estate data to understand property investment patterns.
- Classify properties into good investment and other investment categories.
- Estimate future property values using regression.
- Explore market trends through interactive visualizations.
- Evaluate machine learning models using standard performance metrics.
- Develop an interactive web application for presenting analytical results.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏠 Dashboard | Overview of the real estate dataset and project insights. |
| 📊 Analysis & Prediction | Interactive property investment classification and future value prediction. |
| 📈 Market Insights | Explore property-related patterns and analytical visualizations. |
| 🤖 Model Performance | View classification and regression evaluation metrics. |
| ℹ️ About Project | Understand the project, methodology and technology stack. |

---

## 📂 Dataset Information

The project uses a cleaned real estate dataset for data analysis and machine learning.

| Dataset Information | Details |
|---|---:|
| Total Records | 250,000 |
| Total Features | 43 |
| Data Format | CSV |
| Main Dataset | `cleaned_real_estate_dataset.csv` |

The cleaned dataset is used for exploratory data analysis, preprocessing, model development and application predictions.

---

## 🔍 Exploratory Data Analysis & Preprocessing

The project includes data preparation and analysis stages to make the dataset suitable for machine learning.

**Main workflow:**

- Dataset inspection and understanding.
- Data cleaning and preprocessing.
- Feature preparation for classification and regression.
- Exploratory data analysis.
- Visualization of relevant data patterns.
- Preparation of model inputs using Scikit-learn preprocessing pipelines.

The preprocessing pipelines are saved separately and reused during prediction.

---

## 🤖 Machine Learning Models

Two machine learning approaches are implemented in this project.

### 1. Investment Classification

**Algorithm:** Logistic Regression

The classification model predicts whether a property belongs to the good-investment category based on the project's rule-generated classification target.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 88.742% |
| Precision | 89.13% |
| Recall | 89.50% |
| F1-Score | 89.32% |

**Saved Model Files:**

- `models/good_investment_classifier.pkl`
- `models/classification_preprocessor.pkl`

### 2. Future Property Value Prediction

**Algorithm:** Linear Regression

The regression model estimates future property values using a target generated with a fixed annual appreciation assumption of **8% over five years**.

| Evaluation Metric | Result |
|---|---:|
| RMSE | 0.000868 |
| MAE | 0.000635 |
| R² Score | 1.000000 |

**Saved Model Files:**

- `models/future_price_regressor/`
- `models/regression_preprocessor.pkl`

The regression model is managed using MLflow.

**Important:** The near-perfect R² score reflects the mathematically generated target based on a fixed appreciation formula. It should not be interpreted as proof of real-world property market forecasting accuracy.

---

## 📊 Model Performance

The project evaluates its models using different metrics according to their machine learning tasks.

**Classification Metrics**

- Accuracy
- Precision
- Recall
- F1-Score

**Regression Metrics**

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Evaluation reports are available in the `reports/` directory.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Algorithms | Logistic Regression, Linear Regression |
| Model Management | MLflow |
| Model Persistence | Joblib, Skops |
| Data Visualization | Matplotlib, Seaborn |
| Web Framework | Streamlit |
| Scientific Computing | SciPy |
| Development Environment | Google Colab, Jupyter Notebook |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 🏗️ Project Structure

```text
Real_Estate_Investment_Advisor/
│
├── app/
│   └── app.py
│
├── models/
│   ├── classification_preprocessor.pkl
│   ├── good_investment_classifier.pkl
│   ├── regression_preprocessor.pkl
│   │
│   ├── good_investment_classifier/
│   │   ├── MLmodel
│   │   ├── model.skops
│   │   ├── conda.yaml
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   │
│   └── future_price_regressor/
│       ├── MLmodel
│       ├── model.skops
│       ├── conda.yaml
│       ├── python_env.yaml
│       └── requirements.txt
│
├── reports/
│   ├── classification_model_evaluation.csv
│   ├── preprocessing_report.csv
│   ├── regression_model_evaluation.csv
│   └── eda_plots/
│
├── src/
│   ├── data_preprocessing.py
│   ├── eda_analysis.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── cleaned_real_estate_dataset.csv
├── Real_Estate_Investment_Advisor.ipynb
├── requirements.txt
└── .gitignore
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

👉 [Open Live Application](https://realestateinvestmentadvisor743442piku.streamlit.app/)

The application uses project-relative model paths so that the model artifacts can be loaded in the deployment environment.

---

## ⚠️ Limitations

- The classification target is generated using project-defined rules and does not represent independently verified investment outcomes.
- The future property value target is based on a fixed 8% annual appreciation assumption over five years.
- Regression performance metrics should be interpreted in the context of the simulated target.
- Actual property prices and investment returns can vary due to market conditions, location, economic factors and other variables.
- Predictions are for educational and analytical purposes and should not be treated as guaranteed investment advice.

---

## 🔮 Future Improvements

- Integrate real-time property market data.
- Explore additional machine learning algorithms.
- Improve feature engineering and model validation.
- Incorporate location-based property analysis.
- Develop more advanced investment risk analysis.
- Improve future price estimation using historical market data.

---

## 👨‍💻 Author

**Mriganka Das**

MCA Graduate | Aspiring Data Scientist | AI & Machine Learning Enthusiast

- **GitHub:** [mrigankadas743442-cpu](https://github.com/mrigankadas743442-cpu)
- **Project Repository:** [Real Estate Investment Advisor](https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor)
- **Live Application:** [Real Estate Investment Advisor](https://realestateinvestmentadvisor743442piku.streamlit.app/)
- **LinkedIn:** [Mriganka Das](https://www.linkedin.com/in/mriganka-das-3074bb237)

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning and Streamlit.**