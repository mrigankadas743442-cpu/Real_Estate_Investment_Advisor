
import os
import pandas as pd

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/RealEstateInvestmentAdvisor"

DATA_PATH = os.path.join(
    BASE_DIR, "cleaned_real_estate_dataset.csv"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load processed dataset
# --------------------------------------------------

def load_dataset():
    """Load the feature-engineered real estate dataset."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


# --------------------------------------------------
# 3. Validate required columns
# --------------------------------------------------

def validate_dataset(df):

    required_columns = [
        "ID",
        "City",
        "Property_Type",
        "Price_in_Lakhs",
        "Size_in_SqFt",
        "Good_Investment",
        "Future_Price_5Y"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {missing_columns}"
        )

    print("\nRequired columns validation: PASSED")


# --------------------------------------------------
# 4. Generate data-quality report
# --------------------------------------------------

def generate_report(df):

    report = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Duplicate IDs": int(df["ID"].duplicated().sum()),
        "Classification Target Present":
            "Good_Investment" in df.columns,
        "Regression Target Present":
            "Future_Price_5Y" in df.columns
    }

    report_df = pd.DataFrame(
        list(report.items()),
        columns=["Metric", "Value"]
    )

    report_path = os.path.join(
        REPORT_DIR, "preprocessing_report.csv"
    )

    report_df.to_csv(report_path, index=False)

    print("\nDATA QUALITY REPORT")
    print(report_df.to_string(index=False))

    print("\nReport saved at:")
    print(report_path)


# --------------------------------------------------
# 5. Target variable validation
# --------------------------------------------------

def validate_targets(df):

    print("\nCLASSIFICATION TARGET DISTRIBUTION")

    print(
        df["Good_Investment"]
        .value_counts()
        .sort_index()
    )

    print("\nREGRESSION TARGET SUMMARY")

    print(
        df["Future_Price_5Y"].describe()
    )


# --------------------------------------------------
# 6. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    df = load_dataset()

    validate_dataset(df)

    generate_report(df)

    validate_targets(df)

    print("\nPreprocessing validation completed.")
