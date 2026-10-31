
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/RealEstateInvestmentAdvisor"

DATA_PATH = os.path.join(
    BASE_DIR, "cleaned_real_estate_dataset.csv"
)

PLOT_DIR = os.path.join(
    BASE_DIR, "reports", "eda_plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Dataset loaded:", df.shape)
    return df


# --------------------------------------------------
# 3. Save plot helper
# --------------------------------------------------

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# --------------------------------------------------
# 4. Price and size analysis
# --------------------------------------------------

def price_size_analysis(df):

    # Q1: Property price distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["Price_in_Lakhs"], bins=40, kde=True)
    plt.title("Distribution of Property Prices")
    plt.xlabel("Price (Lakhs)")
    save_plot("01_price_distribution.png")

    # Q2: Property size distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["Size_in_SqFt"], bins=40, kde=True)
    plt.title("Distribution of Property Sizes")
    plt.xlabel("Size (SqFt)")
    save_plot("02_size_distribution.png")

    # Q3: Price per sq ft by property type
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="Property_Type",
        y="Price_per_SqFt_Exact"
    )
    plt.xticks(rotation=30)
    plt.title("Price per SqFt by Property Type")
    save_plot("03_price_by_property_type.png")

    # Q4: Size vs property price
    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        alpha=0.4
    )
    plt.title("Property Size vs Price")
    save_plot("04_size_vs_price.png")


# --------------------------------------------------
# 5. Location-based analysis
# --------------------------------------------------

def location_analysis(df):

    # Q6: Average price per sq ft by state
    state_prices = (
        df.groupby("State")["Price_per_SqFt_Exact"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    state_prices.plot(kind="bar")
    plt.title("Average Price per SqFt by State")
    plt.ylabel("Price per SqFt (Lakhs)")
    save_plot("05_state_price_per_sqft.png")

    # Q7: Average property price by city
    city_prices = (
        df.groupby("City")["Price_in_Lakhs"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 6))
    city_prices.plot(kind="bar")
    plt.title("Top 15 Cities by Average Property Price")
    save_plot("06_city_average_price.png")

    # Q8: Median property age by locality
    locality_age = (
        df.groupby("Locality")["Age_of_Property"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(12, 6))
    locality_age.plot(kind="bar")
    plt.title("Median Property Age by Locality (Top 15)")
    save_plot("07_locality_median_age.png")

    # Q9: BHK distribution across cities
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        x="City",
        hue="BHK"
    )
    plt.xticks(rotation=45)
    plt.title("BHK Distribution Across Cities")
    save_plot("08_bhk_by_city.png")


# --------------------------------------------------
# 6. Feature relationship and correlation
# --------------------------------------------------

def relationship_analysis(df):

    # Q11: Numeric correlation
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0
    )
    plt.title("Numeric Feature Correlation")
    save_plot("09_correlation_heatmap.png")

    # Q12: Nearby schools vs price per sq ft
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Nearby_Schools",
        y="Price_per_SqFt_Exact"
    )
    plt.title("Nearby Schools vs Price per SqFt")
    save_plot("10_schools_vs_price.png")

    # Q13: Nearby hospitals vs price per sq ft
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Nearby_Hospitals",
        y="Price_per_SqFt_Exact"
    )
    plt.title("Nearby Hospitals vs Price per SqFt")
    save_plot("11_hospitals_vs_price.png")

    # Q14: Price by furnished status
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Furnished_Status",
        y="Price_in_Lakhs"
    )
    plt.title("Price by Furnished Status")
    save_plot("12_furnished_status.png")

    # Q15: Price per sq ft by facing
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="Facing",
        y="Price_per_SqFt_Exact"
    )
    plt.title("Price per SqFt by Facing")
    save_plot("13_facing_vs_price.png")


# --------------------------------------------------
# 7. Investment and amenities analysis
# --------------------------------------------------

def investment_analysis(df):

    # Q16: Owner type distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Owner_Type")
    plt.title("Property Distribution by Owner Type")
    save_plot("14_owner_type.png")

    # Q17: Availability status
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="Availability_Status")
    plt.title("Property Availability Status")
    save_plot("15_availability_status.png")

    # Q18: Parking vs property price
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Parking_Space",
        y="Price_in_Lakhs"
    )
    plt.title("Parking Space vs Property Price")
    save_plot("16_parking_vs_price.png")

    # Q19: Amenities vs price per sq ft
    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Amenity_Count",
        y="Price_per_SqFt_Exact"
    )
    plt.title("Amenity Count vs Price per SqFt")
    save_plot("17_amenities_vs_price.png")

    # Q20: Transport accessibility vs investment label
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df,
        x="Public_Transport_Accessibility",
        y="Good_Investment"
    )
    plt.title("Transport Accessibility vs Good Investment Rate")
    plt.ylabel("Good Investment Rate")
    save_plot("18_transport_vs_investment.png")


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    df = load_data()

    price_size_analysis(df)
    location_analysis(df)
    relationship_analysis(df)
    investment_analysis(df)

    print("\nEDA analysis completed successfully!")
    print("Total plots generated: 18")
    print("Plots saved in:", PLOT_DIR)
