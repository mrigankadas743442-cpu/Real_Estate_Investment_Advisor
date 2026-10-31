
import os
import mlflow
import pandas as pd

# Project paths
BASE_DIR = "/content/drive/MyDrive/RealEstateInvestmentAdvisor"
REPORT_DIR = os.path.join(BASE_DIR, "reports")
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

os.makedirs(REPORT_DIR, exist_ok=True)

# Connect to existing MLflow database
mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB)

experiment = mlflow.get_experiment_by_name(
    "Real_Estate_Investment_Advisor_Persistent"
)

if experiment is None:
    raise ValueError("MLflow experiment not found.")

# Existing successful model run IDs
classification_runs = {
    "Logistic Regression": "0e887e6ad2de441e8932119b66c3206f",
    "Decision Tree": "a33f695dfe554178991cedc8f4bec1e0",
    "Random Forest": "7099ed8709674e1cb69eed5e03bb7da8",
    "XGBoost": "8d5079c7dc4c40bc81a59e9a1e57142a"
}

regression_runs = {
    "Linear Regression": "7bc933044958433d813936151d3b6108",
    "Decision Tree": "4566fdf8a5764e088c1c276e26e8327d",
    "Random Forest": "82ffee7420294334a30bc942bfbd439b",
    "XGBoost": "6cf632336cfc43f4b638aad577414a7d"
}

# Retrieve existing MLflow runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    output_format="pandas"
)

# Classification evaluation
classification_rows = []

for model_name, run_id in classification_runs.items():
    match = runs[runs["run_id"] == run_id]

    if match.empty:
        print(f"Warning: {model_name} run not found.")
        continue

    run = match.iloc[0]

    classification_rows.append({
        "Model": model_name,
        "Accuracy": run.get("metrics.accuracy"),
        "Precision": run.get("metrics.precision"),
        "Recall": run.get("metrics.recall"),
        "F1_Score": run.get("metrics.f1_score")
    })

classification_df = pd.DataFrame(classification_rows)

# Regression evaluation
regression_rows = []

for model_name, run_id in regression_runs.items():
    match = runs[runs["run_id"] == run_id]

    if match.empty:
        print(f"Warning: {model_name} run not found.")
        continue

    run = match.iloc[0]

    regression_rows.append({
        "Model": model_name,
        "RMSE": run.get("metrics.RMSE"),
        "MAE": run.get("metrics.MAE"),
        "R2_Score": run.get("metrics.R2_Score")
    })

regression_df = pd.DataFrame(regression_rows)

# Save reports
classification_path = os.path.join(
    REPORT_DIR, "classification_model_evaluation.csv"
)

regression_path = os.path.join(
    REPORT_DIR, "regression_model_evaluation.csv"
)

classification_df.to_csv(classification_path, index=False)
regression_df.to_csv(regression_path, index=False)

# Display results
print("\nCLASSIFICATION MODEL EVALUATION")
print(classification_df.to_string(index=False))

print("\nREGRESSION MODEL EVALUATION")
print(regression_df.to_string(index=False))

print("\nEvaluation reports saved successfully:")
print(classification_path)
print(regression_path)
