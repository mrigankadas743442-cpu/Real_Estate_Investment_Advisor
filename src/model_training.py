
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/RealEstateInvestmentAdvisor"

DATA_PATH = os.path.join(
    BASE_DIR, "cleaned_real_estate_dataset.csv"
)

MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")

TRACKING_URI = "sqlite:///" + MLFLOW_DB

EXPERIMENT_NAME = "Real_Estate_Investment_Advisor_Persistent"

# Safety: training is disabled by default to prevent
# accidental duplicate MLflow runs.
RUN_MODEL_TRAINING = (
    os.getenv("RUN_MODEL_TRAINING", "0") == "1"
)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# --------------------------------------------------
# 3. ORIGINAL FEATURE SELECTION
# --------------------------------------------------

classification_features = [
    'State', 'City', 'Locality', 'Property_Type',
    'Size_in_SqFt', 'Furnished_Status',
    'Age_of_Property', 'Parking_Space',
    'Security', 'Facing', 'Owner_Type',
    'Amenity_Pool', 'Amenity_Gym',
    'Amenity_Garden', 'Amenity_Clubhouse',
    'Amenity_Playground'
]

regression_features = [
    'State', 'City', 'Locality', 'Property_Type',
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs',
    'Furnished_Status', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals',
    'Public_Transport_Accessibility',
    'Parking_Space', 'Security', 'Facing',
    'Owner_Type', 'Availability_Status',
    'Amenity_Pool', 'Amenity_Gym',
    'Amenity_Garden', 'Amenity_Clubhouse',
    'Amenity_Playground', 'School_Score',
    'Hospital_Score', 'Transport_Score',
    'Infrastructure_Score'
]

X_cls = df[classification_features]
y_cls = df["Good_Investment"]

X_reg = df[regression_features]
y_reg = df["Future_Price_5Y"]

# --------------------------------------------------
# 4. ORIGINAL TRAIN-TEST SPLIT
# --------------------------------------------------

X_train_cls, X_test_cls, y_train_cls, y_test_cls = (
    train_test_split(
        X_cls,
        y_cls,
        test_size=0.20,
        random_state=42,
        stratify=y_cls
    )
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

print("Classification split:", X_train_cls.shape, X_test_cls.shape)
print("Regression split:", X_train_reg.shape, X_test_reg.shape)

# --------------------------------------------------
# 5. ORIGINAL PREPROCESSING
# --------------------------------------------------

def create_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )


classification_preprocessor = create_preprocessor(X_train_cls)
regression_preprocessor = create_preprocessor(X_train_reg)

X_train_cls_processed = classification_preprocessor.fit_transform(
    X_train_cls
)
X_test_cls_processed = classification_preprocessor.transform(
    X_test_cls
)

X_train_reg_processed = regression_preprocessor.fit_transform(
    X_train_reg
)
X_test_reg_processed = regression_preprocessor.transform(
    X_test_reg
)

print("Classification processed:", X_train_cls_processed.shape)
print("Regression processed:", X_train_reg_processed.shape)

# --------------------------------------------------
# 6. ORIGINAL CLASSIFICATION MODELS
# --------------------------------------------------

classification_models = {
    "Logistic_Regression_Baseline": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision_Tree_Classifier_Final": DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random_Forest_Classifier": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost_Classifier": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 7. ORIGINAL REGRESSION MODELS
# --------------------------------------------------

regression_models = {
    "Linear_Regression_Baseline": LinearRegression(),

    "Decision_Tree_Regressor": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random_Forest_Regressor_Retry": RandomForestRegressor(
        n_estimators=30,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost_Regressor": XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------------
# 8. TRAINING AND MLFLOW TRACKING
# --------------------------------------------------

def train_classification_models():

    results = []

    for run_name, model in classification_models.items():

        print("\nTraining:", run_name)

        with mlflow.start_run(run_name=run_name) as run:

            model.fit(
                X_train_cls_processed,
                y_train_cls
            )

            predictions = model.predict(
                X_test_cls_processed
            )

            metrics = {
                "accuracy": accuracy_score(y_test_cls, predictions),
                "precision": precision_score(
                    y_test_cls, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_test_cls, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_test_cls, predictions, zero_division=0
                )
            }

            mlflow.log_param("model", run_name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                name=run_name.lower(),
                serialization_format="pickle"
            )

            results.append({
                "Model": run_name,
                **metrics,
                "Run_ID": run.info.run_id
            })

            print(metrics)
            print("Run ID:", run.info.run_id)

    return pd.DataFrame(results)


def train_regression_models():

    results = []

    for run_name, model in regression_models.items():

        print("\nTraining:", run_name)

        with mlflow.start_run(run_name=run_name) as run:

            model.fit(
                X_train_reg_processed,
                y_train_reg
            )

            predictions = model.predict(
                X_test_reg_processed
            )

            metrics = {
                "RMSE": np.sqrt(
                    mean_squared_error(y_test_reg, predictions)
                ),
                "MAE": mean_absolute_error(
                    y_test_reg, predictions
                ),
                "R2_Score": r2_score(
                    y_test_reg, predictions
                )
            }

            mlflow.log_param("model", run_name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", 42)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=model,
                name=run_name.lower(),
                serialization_format="pickle"
            )

            results.append({
                "Model": run_name,
                **metrics,
                "Run_ID": run.info.run_id
            })

            print(metrics)
            print("Run ID:", run.info.run_id)

    return pd.DataFrame(results)


# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    if not RUN_MODEL_TRAINING:
        print("\nTraining was not started.")
        print("Existing MLflow runs are preserved.")
        print(
            "To intentionally reproduce training, set "
            "RUN_MODEL_TRAINING=1."
        )

    else:

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        os.makedirs(
            os.path.join(BASE_DIR, "reports"),
            exist_ok=True
        )

        classification_results = train_classification_models()
        regression_results = train_regression_models()

        classification_results.to_csv(
            os.path.join(
                BASE_DIR, "reports",
                "classification_training_results.csv"
            ),
            index=False
        )

        regression_results.to_csv(
            os.path.join(
                BASE_DIR, "reports",
                "regression_training_results.csv"
            ),
            index=False
        )

        print("\nAll model training workflows completed.")
