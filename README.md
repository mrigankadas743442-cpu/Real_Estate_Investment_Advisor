🏠 Real Estate Investment Advisor

Predicting Property Investment Potential and Future Value with Machine Learning

""Live Demo" (https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)" (https://realestateinvestmentadvisor743442piku.streamlit.app/)
""Python" (https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)" (https://www.python.org/)
""Streamlit" (https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)" (https://streamlit.io/)
""Scikit-learn" (https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)" (https://scikit-learn.org/)
""MLflow" (https://img.shields.io/badge/MLflow-Model%20Lifecycle-0194E2)" (https://mlflow.org/)
""Pandas" (https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)" (https://pandas.pydata.org/)
""NumPy" (https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)" (https://numpy.org/)
""XGBoost" (https://img.shields.io/badge/XGBoost-Machine%20Learning-EC6B23)" (https://xgboost.readthedocs.io/)
""Matplotlib" (https://img.shields.io/badge/Matplotlib-Visualization-11557C)" (https://matplotlib.org/)
""Seaborn" (https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0)" (https://seaborn.pydata.org/)
""Joblib" (https://img.shields.io/badge/Joblib-Model%20Persistence-6A5ACD)" (https://joblib.readthedocs.io/)
""GitHub" (https://img.shields.io/badge/GitHub-Version%20Control-181717?logo=github&logoColor=white)" (https://github.com/)
""Deployment" (https://img.shields.io/badge/Deployment-Streamlit%20Cloud-FF4B4B)" (https://share.streamlit.io/)
""Data Science" (https://img.shields.io/badge/Data%20Science-Analytics-2E8B57)" (#project-overview)
""Machine Learning" (https://img.shields.io/badge/Machine%20Learning-Classification%20%7C%20Regression-orange)" (#machine-learning-models)

---

📌 Project Overview

Real Estate Investment Advisor is an end-to-end Data Science and Machine Learning project designed to analyze real-estate properties, evaluate investment potential, and estimate future property values.

The project combines data preprocessing, exploratory data analysis, machine learning, model evaluation, and an interactive Streamlit web application.

The cleaned dataset contains 250,000 records and 43 columns.

The application provides two main machine-learning capabilities:

- Investment Classification: Predicts whether a property meets the project's investment classification criteria.
- Future Property Value Prediction: Estimates future property value using a trained regression model.

🚀 Live Application: "Open Real Estate Investment Advisor" (https://realestateinvestmentadvisor743442piku.streamlit.app/)

💻 GitHub Repository: "View Source Code" (https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor)

---

✨ Application Features

The Streamlit application contains five main sections.

Section| Description
Dashboard / Home| Displays dataset overview and key real-estate metrics.
Analysis & Prediction| Accepts property details and generates machine-learning predictions.
Market Insights| Provides exploratory data analysis and market visualizations.
Model Performance| Displays classification and regression evaluation metrics.
About Project| Provides project information, objectives, and implementation details.

---

🤖 Machine Learning Models

1. Investment Classification

- Algorithm: Logistic Regression
- Task: Binary classification.
- Objective: Classify properties according to the project's investment target.
- Preprocessing: Scikit-learn ColumnTransformer.
- Model Persistence: Joblib.

Saved artifacts:

- "good_investment_classifier.pkl"
- "classification_preprocessor.pkl"

2. Future Property Value Prediction

- Algorithm: Linear Regression
- Task: Regression.
- Objective: Estimate future property value over a five-year period.
- Preprocessing: Scikit-learn ColumnTransformer.
- Model Management: MLflow.

Saved artifacts:

- "future_price_regressor/"
- "regression_preprocessor.pkl"

3. Model Management

MLflow is used to organize and manage trained model artifacts. Model configuration files and environment metadata are included in the repository.

---

📊 Model Performance

Classification — Logistic Regression

Evaluation Metric| Score
Accuracy| 88.742%
Precision| 89.13%
Recall| 89.50%
F1-Score| 89.32%

Regression — Linear Regression

Evaluation Metric| Score
RMSE| 0.000868
MAE| 0.000635
R² Score| 1.000000

Important: The regression target was generated using a fixed 8% annual appreciation assumption over five years. Therefore, the near-perfect R² indicates that the model reproduces this simulated formula; it does not demonstrate equivalent accuracy in predicting actual real-estate market prices.

The classification target is also rule-generated rather than based on independently verified investment outcomes. These metrics should be interpreted in this context.

---

🛠️ Technology Stack

Category| Technologies
Programming Language| Python
Web Framework| Streamlit
Machine Learning| Scikit-learn, XGBoost
Deep Data Processing| Pandas, NumPy
Model Management| MLflow
Model Persistence| Joblib, Skops
Data Visualization| Matplotlib, Seaborn
Scientific Computing| SciPy
Development Environment| Google Colab, Jupyter Notebook
Version Control| Git, GitHub
Deployment| Streamlit Community Cloud

---

📁 Project Structure

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

---

⚙️ Installation and Local Setup

Prerequisites

- Python 3.12 recommended.
- Git installed.
- Compatible Python environment for the required machine-learning libraries.

Step 1: Clone the repository

git clone https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor.git

cd Real_Estate_Investment_Advisor

Step 2: Create a virtual environment

Windows PowerShell:

python -m venv .venv

.venv\Scripts\Activate.ps1

macOS / Linux:

python3 -m venv .venv

source .venv/bin/activate

Step 3: Install dependencies

python -m pip install --upgrade pip

pip install -r requirements.txt

Step 4: Run the Streamlit application

From the project root directory, execute:

streamlit run app/app.py

The application will be available at:

http://localhost:8501

Important Environment Compatibility

The saved model artifacts were created using these core library versions:

- Scikit-learn 1.6.1
- NumPy 2.1.3
- Pandas 2.2.3
- MLflow 3.17.2
- Skops 0.16.0

The root "requirements.txt" pins these core versions. Use a compatible Python version and test model loading after changing dependency versions.

---

🌐 Live Application

The project is deployed using Streamlit Community Cloud.

Access the application here:

👉 "Real Estate Investment Advisor — Live Demo" (https://realestateinvestmentadvisor743442piku.streamlit.app/)

How to Use

1. Open the live application.
2. Explore the Dashboard / Home section.
3. Navigate to Analysis & Prediction.
4. Enter the available property details.
5. Review the investment classification and future-value prediction.
6. Explore Market Insights and Model Performance.

---

⚠️ Limitations and Responsible Use

- The classification target is rule-generated and is not based on independently verified investment returns.
- The regression target uses a fixed 8% annual appreciation assumption over five years.
- Real-estate prices depend on location, infrastructure, demand, supply, interest rates, property condition, and other market factors.
- The model metrics describe performance against the project's prepared targets and test data.
- Predictions do not guarantee future property values or investment returns.

This application is intended for educational, analytical, and demonstration purposes. It should not replace professional property valuation, financial advice, or independent due diligence.

---

🔮 Future Improvements

- Evaluate models using independently sourced and verified property transaction data.
- Improve prediction reliability across different geographic regions.
- Experiment with additional classification and regression algorithms.
- Implement model explainability techniques.
- Improve automated testing and deployment monitoring.
- Introduce automated dataset updates and model-version tracking.

---

👨‍💻 Author

Mriganka Das

MCA Graduate | Aspiring Data Science & AI/ML Professional

- GitHub Profile: "mrigankadas743442-cpu" (https://github.com/mrigankadas743442-cpu)
- Project Repository: "Real Estate Investment Advisor" (https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor)
- Live Application: "Streamlit Demo" (https://realestateinvestmentadvisor743442piku.streamlit.app/)

---

⭐ If you find this project useful, feel free to explore the repository and try the live application.