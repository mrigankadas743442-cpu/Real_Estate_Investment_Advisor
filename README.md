Real Estate Investment Advisor
� � � � � � � � � � � � � �
Predicting Property Investment Potential and Future Value with Machine Learning
� � � �
An end-to-end machine learning project that analyzes real-estate data, classifies properties according to the project's investment criteria, and estimates future property value. It combines data preprocessing, exploratory data analysis, model training, evaluation, and an interactive Streamlit application.
Live Demo: https://realestateinvestmentadvisor743442piku.streamlit.app/
GitHub Repository: https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor

Project Overview
Real-estate investment analysis involves evaluating property characteristics, exploring market patterns, and estimating potential future value. This project demonstrates how data science and machine learning can support these activities through an interactive application.
The cleaned dataset contains 250,000 records and 43 columns. The project includes: - A classification workflow to predict whether a property meets the project's investment classification criteria. - A regression workflow to estimate future property value using the project's five-year target. - An interactive dashboard for exploring property data, insights, predictions, and model evaluation.
This is an educational and analytical project. Its predictions should not be interpreted as financial or investment advice.
Application Features
The Streamlit application contains five main sections:
Section                             Description
Dashboard / Home                Displays an overview of the dataset and key property-market metrics.
Analysis & Prediction           Accepts property inputs and runs the trained classification and regression workflows.
Market Insights                 Presents exploratory analysis and visual insights from the real-estate data.
Model Performance               Displays evaluation results for the trained machine-learning models.
About Project                   Describes the project objectives and implementation.
Machine Learning Models
Investment Classification
Algorithm: Logistic Regression
Task: Binary classification based on the project's investment target.
Preprocessing: Saved Scikit-learn ColumnTransformer.
Artifacts: models/good_investment_classifier.pkl and models/classification_preprocessor.pkl.
Future Property Value Regression
Algorithm: Linear Regression
Task: Estimate future property value using the project's five-year target.
Preprocessing: Saved Scikit-learn ColumnTransformer.
Model format: MLflow model.
Artifacts: models/future_price_regressor/ and models/regression_preprocessor.pkl.
Experiment and Model Management
MLflow model files and environment metadata are included in the repository to organize and load the trained models.
Model Performance
The recorded project evaluation results are:
Classification --- Logistic Regression
Metric         Result
Accuracy      88.742% Precision      89.13% Recall         89.50% F1-score       89.32%
Regression --- Linear Regression
Metric       Result
RMSE       0.000868 MAE        0.000635 R²         1.000000
Important interpretation: The regression target was generated using a fixed 8% annual appreciation assumption over five years. The near-perfect R² therefore indicates that the model reproduces this simulated formula; it does not establish equivalent accuracy in forecasting actual property-market prices. The classification target is also rule-generated rather than based on independently verified investment outcomes. See Limitations and Responsible Use.
Technology Stack
Area                Technologies
Programming         Python Data manipulation   Pandas, NumPy Machine learning    Scikit-learn, XGBoost Model persistence   Joblib, Skops Model lifecycle     MLflow Visualization       Matplotlib, Seaborn Web application     Streamlit Development         Google Colab, Jupyter Notebook Version control     Git, GitHub Deployment          Streamlit Community Cloud
Project Structure
Real_Estate_Investment_Advisor/
│
├── app/
│   └── app.py
│
├── models/
│   ├── classification_preprocessor.pkl
│   ├── good_investment_classifier.pkl
│   ├── regression_preprocessor.pkl
│   ├── good_investment_classifier/
│   │   ├── MLmodel
│   │   ├── model.skops
│   │   ├── conda.yaml
│   │   ├── python_env.yaml
│   │   └── requirements.txt
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
Installation and Local Setup
Prerequisites
Python 3.12 is recommended to align with the deployed environment.
Git.
A compatible Python environment for the pinned machine-learning dependencies.
1. Clone the repository
git clone https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor.git
cd Real_Estate_Investment_Advisor
2. Create and activate a virtual environment
Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Run the Streamlit application
Run from the repository root:
streamlit run app/app.py
Streamlit will print a local address, usually http://localhost:8501.
Environment compatibility
The saved model artifacts were created with these core package versions: - Scikit-learn 1.6.1 - NumPy 2.1.3 - Pandas 2.2.3 - MLflow 3.17.2 - Skops 0.16.0
The root requirements.txt pins these core versions. Use a compatible Python version and test model loading after changing dependency versions.
How to Use
Open the live application.
Explore the Dashboard / Home section.
Open Analysis & Prediction and enter the available property details.
Review the classification and future-value outputs.
Explore Market Insights and Model Performance to understand the dataset and evaluation results.
Limitations and Responsible Use
The classification target is rule-generated; it is not a label based on independently verified investment returns.
The regression target uses a fixed 8% annual appreciation assumption over five years. The model learns a simulated relationship rather than independently discovering future market movements.
Real property values depend on location, supply and demand, interest rates, infrastructure, regulation, property condition, and other factors that may not be fully represented in the dataset.
Evaluation metrics describe performance against the project's prepared targets and test data. They do not guarantee future results.
This application is an educational and analytical demonstration, not a substitute for professional property valuation, financial advice, or independent due diligence.
Future Improvements
Evaluate models using independently sourced and verified property transaction data.
Validate performance across geographic regions and different market periods.
Compare additional algorithms using robust cross-validation.
Add model explainability to show which property features influence predictions.
Improve automated testing, input validation, and deployment monitoring.
Add data-refresh workflows and model-version tracking.
Author
Mriganka Das
MCA Graduate | Aspiring Data Science & AI/ML Professional
GitHub: https://github.com/mrigankadas743442-cpu
Project Repository: https://github.com/mrigankadas743442-cpu/Real_Estate_Investment_Advisor
Live Demo: https://realestateinvestmentadvisor743442piku.streamlit.app/
If you find this project useful, explore the repository and try the live application.