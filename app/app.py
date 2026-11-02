



import os
import joblib
import mlflow.pyfunc
import pandas as pd
import streamlit as st


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATA_PATH = os.path.join(
    BASE_DIR, "cleaned_real_estate_dataset.csv"
)

CLASSIFIER_PATH = os.path.join(
    MODEL_DIR, "good_investment_classifier.pkl"
)

CLASSIFICATION_PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR, "classification_preprocessor.pkl"
)

REGRESSION_PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR, "regression_preprocessor.pkl"
)

REGRESSION_MODEL_PATH = os.path.join(
    MODEL_DIR, "future_price_regressor"
)


# ==================================================
# 2. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)


# ==================================================
# 3. LOAD MODELS AND DATASET
# ==================================================

@st.cache_resource
def load_models():

    classifier = joblib.load(CLASSIFIER_PATH)

    classification_preprocessor = joblib.load(
        CLASSIFICATION_PREPROCESSOR_PATH
    )

    regression_preprocessor = joblib.load(
        REGRESSION_PREPROCESSOR_PATH
    )

    regressor = mlflow.pyfunc.load_model(
        REGRESSION_MODEL_PATH
    )

    return (
        classifier,
        classification_preprocessor,
        regressor,
        regression_preprocessor
    )


@st.cache_data
def load_dataset():

    data = pd.read_csv(DATA_PATH)

    for column in ["State", "City", "Locality"]:
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(
        subset=["State", "City", "Locality"]
    )

    return data


try:

    (
        classifier,
        classification_preprocessor,
        regressor,
        regression_preprocessor
    ) = load_models()

    df = load_dataset()

except Exception as error:

    st.error(f"Error loading project files: {error}")
    st.stop()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def category_options(column):

    values = df[column].dropna().unique().tolist()

    return sorted(values, key=lambda value: str(value))


def numeric_bounds(column):

    values = pd.to_numeric(
        df[column], errors="coerce"
    ).dropna()

    if values.empty:
        raise ValueError(
            f"No valid training values found for {column}"
        )

    minimum = float(values.min())
    maximum = float(values.max())
    median = float(values.median())

    if minimum == maximum:
        maximum = minimum + 1.0

    return minimum, maximum, median


def score_input(label, column, key):

    minimum, maximum, median = numeric_bounds(column)

    step = max((maximum - minimum) / 100, 0.001)

    return st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        value=median,
        step=step,
        key=key
    )


# ==================================================

# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

with st.sidebar:
    st.title("🏠 Property Advisor")
    st.caption("Real Estate Analytics Platform")
    st.divider()

    selected_page = st.radio(
        "MAIN MENU",
        [
            "Dashboard / Home",
            "Analysis & Prediction",
            "Market Insights",
            "Model Performance",
            "About Project"
        ],
        index=0
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Version 1.0")


# ==================================================
# 6. DASHBOARD / HOME
# ==================================================

if selected_page == "Dashboard / Home":

    st.title("🏠 Real Estate Investment Advisor")
    st.subheader("Dashboard Overview")
    st.write(
        "Explore property investment potential, market patterns, "
        "and machine-learning model performance."
    )
    st.divider()

    total_properties = len(df)
    total_states = df["State"].nunique()
    total_cities = df["City"].nunique()
    average_price = df["Price_in_Lakhs"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Properties", f"{total_properties:,}")
    k2.metric("States Covered", f"{total_states:,}")
    k3.metric("Cities Covered", f"{total_cities:,}")
    k4.metric("Average Listed Price", f"₹{average_price:.2f} L")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Cities by Average Property Price")
        city_prices = (
            df.groupby("City")["Price_in_Lakhs"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        st.bar_chart(city_prices, horizontal=True)

    with right:
        st.subheader("Investment Label Distribution")
        label_counts = (
            df["Good_Investment"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )
        label_counts.index = [
            "Not a Good Investment",
            "Good Investment"
        ]
        st.bar_chart(label_counts)

    st.divider()
    st.subheader("Explore the platform")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Analysis & Prediction**")
        st.write("Evaluate an individual property's investment label and simulated five-year value.")
    with c2:
        st.markdown("**Market Insights**")
        st.write("Explore prices by city, property type, BHK and location.")
    with c3:
        st.markdown("**Model Performance**")
        st.write("Review classification and regression evaluation metrics.")

    st.info(
        "Project limitation: Good Investment labels are generated from "
        "project-defined rules. The five-year price target uses a fixed "
        "8% annual appreciation assumption; it is not a verified market forecast."
    )


# ==================================================
# 7. ANALYSIS & PREDICTION
# ==================================================

elif selected_page == "Analysis & Prediction":

    # 5. APPLICATION HEADER
    # ==================================================

    st.title("🏠 Real Estate Investment Advisor")

    st.subheader(
        "Predict Property Investment Potential & Future Value"
    )

    st.write(
        "Enter property details to evaluate investment potential "
        "and estimate its price after five years."
    )

    st.divider()


    # ==================================================
    # 6. PROPERTY LOCATION
    # ==================================================

    st.header("Property Details")

    col1, col2, col3 = st.columns(3)

    with col1:

        state_options = category_options("State")

        state = st.selectbox(
            "State",
            state_options,
            key="state_select"
        )

        # Cities belonging to selected state only
        city_options = sorted(
            df.loc[
                df["State"] == state,
                "City"
            ].dropna().unique().tolist()
        )

        if not city_options:
            st.error("No cities found for the selected state.")
            st.stop()

        city = st.selectbox(
            "City",
            city_options,
            key=f"city_select_{state}"
        )

        # Localities belonging to selected state and city
        locality_options = sorted(
            df.loc[
                (df["State"] == state) &
                (df["City"] == city),
                "Locality"
            ].dropna().unique().tolist()
        )

        if not locality_options:
            st.error("No localities found for this city.")
            st.stop()

        locality = st.selectbox(
            "Locality",
            locality_options,
            key=f"locality_select_{state}_{city}"
        )

        property_type = st.selectbox(
            "Property Type",
            category_options("Property_Type")
        )

        bhk = st.selectbox(
            "BHK",
            sorted(df["BHK"].dropna().unique().tolist())
        )


    # ==================================================
    # 7. PROPERTY CHARACTERISTICS
    # ==================================================

    with col2:

        size_min, size_max, size_median = numeric_bounds(
            "Size_in_SqFt"
        )

        size = st.number_input(
            "Property Size (SqFt)",
            min_value=size_min,
            max_value=size_max,
            value=size_median,
            step=1.0
        )

        price_min, price_max, price_median = numeric_bounds(
            "Price_in_Lakhs"
        )

        price = st.number_input(
            "Current Property Price (Lakhs)",
            min_value=price_min,
            max_value=price_max,
            value=price_median,
            step=1.0
        )

        age_min, age_max, age_median = numeric_bounds(
            "Age_of_Property"
        )

        age = st.number_input(
            "Property Age (Years)",
            min_value=age_min,
            max_value=age_max,
            value=age_median,
            step=1.0
        )

        furnished = st.selectbox(
            "Furnished Status",
            category_options("Furnished_Status")
        )

        availability = st.selectbox(
            "Availability Status",
            category_options("Availability_Status")
        )


    # ==================================================
    # 8. ADDITIONAL PROPERTY FEATURES
    # ==================================================

    with col3:

        parking = st.selectbox(
            "Parking Space",
            category_options("Parking_Space")
        )

        security = st.selectbox(
            "Security",
            category_options("Security")
        )

        facing = st.selectbox(
            "Facing",
            category_options("Facing")
        )

        owner_type = st.selectbox(
            "Owner Type",
            category_options("Owner_Type")
        )

        transport = st.selectbox(
            "Public Transport Accessibility",
            category_options("Public_Transport_Accessibility")
        )


    # ==================================================
    # 9. NEARBY FACILITIES
    # ==================================================

    st.divider()
    st.subheader("Nearby Facilities")

    facility_col1, facility_col2 = st.columns(2)

    with facility_col1:

        schools_min, schools_max, schools_median = numeric_bounds(
            "Nearby_Schools"
        )

        schools = st.number_input(
            "Nearby Schools",
            min_value=schools_min,
            max_value=schools_max,
            value=schools_median,
            step=1.0
        )

    with facility_col2:

        hospitals_min, hospitals_max, hospitals_median = numeric_bounds(
            "Nearby_Hospitals"
        )

        hospitals = st.number_input(
            "Nearby Hospitals",
            min_value=hospitals_min,
            max_value=hospitals_max,
            value=hospitals_median,
            step=1.0
        )


    # ==================================================
    # 10. AMENITIES
    # ==================================================

    st.divider()
    st.subheader("Property Amenities")

    amenity_col1, amenity_col2, amenity_col3 = st.columns(3)

    with amenity_col1:

        pool = st.checkbox("Swimming Pool")
        gym = st.checkbox("Gym")

    with amenity_col2:

        garden = st.checkbox("Garden")
        clubhouse = st.checkbox("Clubhouse")

    with amenity_col3:

        playground = st.checkbox("Playground")


    # ==================================================
    # 11. INFRASTRUCTURE SCORES
    # ==================================================

    st.divider()
    st.subheader("Infrastructure Scores")

    st.caption(
        "Score ranges and default values are taken from "
        "the cleaned training dataset."
    )

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        school_score = score_input(
            "School Score",
            "School_Score",
            "school_score_input"
        )

        hospital_score = score_input(
            "Hospital Score",
            "Hospital_Score",
            "hospital_score_input"
        )

    with score_col2:

        transport_score = score_input(
            "Transport Score",
            "Transport_Score",
            "transport_score_input"
        )

        infrastructure_score = score_input(
            "Infrastructure Score",
            "Infrastructure_Score",
            "infrastructure_score_input"
        )


    # ==================================================
    # 12. ANALYZE PROPERTY
    # ==================================================

    st.divider()

    submitted = st.button(
        "Analyze Investment",
        use_container_width=True,
        type="primary"
    )


    # ==================================================
    # 13. PREDICTION
    # ==================================================

    if submitted:

        try:

            # ------------------------------------------
            # PREPARE INPUT VALUES
            # ------------------------------------------

            amenity_pool = int(pool)
            amenity_gym = int(gym)
            amenity_garden = int(garden)
            amenity_clubhouse = int(clubhouse)
            amenity_playground = int(playground)

            exact_price_per_sqft = price / size

            # ==========================================
            # CLASSIFICATION: 21 FEATURES
            # ==========================================

            classification_input = pd.DataFrame([{

                "State": state,
                "City": city,
                "Locality": locality,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_in_Lakhs": price,
                "Price_per_SqFt_Exact": exact_price_per_sqft,
                "Furnished_Status": furnished,
                "Age_of_Property": age,
                "Parking_Space": parking,
                "Security": security,
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability,
                "Amenity_Pool": amenity_pool,
                "Amenity_Gym": amenity_gym,
                "Amenity_Garden": amenity_garden,
                "Amenity_Clubhouse": amenity_clubhouse,
                "Amenity_Playground": amenity_playground,
                "Infrastructure_Score": infrastructure_score

            }])

            classification_features = [

                "State",
                "City",
                "Locality",
                "Property_Type",
                "BHK",
                "Size_in_SqFt",
                "Price_in_Lakhs",
                "Price_per_SqFt_Exact",
                "Furnished_Status",
                "Age_of_Property",
                "Parking_Space",
                "Security",
                "Facing",
                "Owner_Type",
                "Availability_Status",
                "Amenity_Pool",
                "Amenity_Gym",
                "Amenity_Garden",
                "Amenity_Clubhouse",
                "Amenity_Playground",
                "Infrastructure_Score"

            ]

            classification_input = classification_input[
                classification_features
            ]

            classification_processed = (
                classification_preprocessor.transform(
                    classification_input
                )
            )

            prediction = classifier.predict(
                classification_processed
            )[0]

            probabilities = classifier.predict_proba(
                classification_processed
            )[0]

            class_list = list(classifier.classes_)

            good_index = class_list.index(1)
            not_good_index = class_list.index(0)

            good_probability = probabilities[good_index] * 100
            not_good_probability = probabilities[not_good_index] * 100

            predicted_index = class_list.index(prediction)

            confidence = probabilities[predicted_index] * 100

            # ==========================================
            # REGRESSION: 26 FEATURES
            # ==========================================

            regression_input = pd.DataFrame([{

                "State": state,
                "City": city,
                "Locality": locality,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size,
                "Price_in_Lakhs": price,
                "Furnished_Status": furnished,
                "Age_of_Property": age,
                "Nearby_Schools": schools,
                "Nearby_Hospitals": hospitals,
                "Public_Transport_Accessibility": transport,
                "Parking_Space": parking,
                "Security": security,
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability,
                "Amenity_Pool": amenity_pool,
                "Amenity_Gym": amenity_gym,
                "Amenity_Garden": amenity_garden,
                "Amenity_Clubhouse": amenity_clubhouse,
                "Amenity_Playground": amenity_playground,
                "School_Score": school_score,
                "Hospital_Score": hospital_score,
                "Transport_Score": transport_score,
                "Infrastructure_Score": infrastructure_score

            }])

            regression_features = [

                "State",
                "City",
                "Locality",
                "Property_Type",
                "BHK",
                "Size_in_SqFt",
                "Price_in_Lakhs",
                "Furnished_Status",
                "Age_of_Property",
                "Nearby_Schools",
                "Nearby_Hospitals",
                "Public_Transport_Accessibility",
                "Parking_Space",
                "Security",
                "Facing",
                "Owner_Type",
                "Availability_Status",
                "Amenity_Pool",
                "Amenity_Gym",
                "Amenity_Garden",
                "Amenity_Clubhouse",
                "Amenity_Playground",
                "School_Score",
                "Hospital_Score",
                "Transport_Score",
                "Infrastructure_Score"

            ]

            regression_input = regression_input[
                regression_features
            ]

            regression_processed = (
                regression_preprocessor.transform(
                    regression_input
                )
            )

            future_price = regressor.predict(
                regression_processed
            )[0]

            future_price = float(future_price)

            # ==========================================
            # DISPLAY CLASSIFICATION RESULTS
            # ==========================================

            st.divider()
            st.header("Investment Analysis Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.subheader("Investment Classification")

                if prediction == 1:

                    st.success("Good Investment")

                else:

                    st.warning("Not a Good Investment")

                st.metric(
                    "Predicted Class Confidence",
                    f"{confidence:.2f}%"
                )

            with result_col2:

                st.subheader("Estimated Future Property Value")

                st.metric(
                    "Current Property Price",
                    f"₹ {price:.2f} Lakhs"
                )

                st.metric(
                    "Estimated Price After 5 Years",
                    f"₹ {future_price:.2f} Lakhs"
                )

            # ==========================================
            # SHOW BOTH CLASS PROBABILITIES
            # ==========================================

            st.subheader("Classification Probability")

            probability_col1, probability_col2 = st.columns(2)

            with probability_col1:

                st.metric(
                    "Good Investment",
                    f"{good_probability:.2f}%"
                )

                st.progress(
                    float(good_probability / 100)
                )

            with probability_col2:

                st.metric(
                    "Not a Good Investment",
                    f"{not_good_probability:.2f}%"
                )

                st.progress(
                    float(not_good_probability / 100)
                )

            # ==========================================
            # PROPERTY SUMMARY
            # ==========================================

            st.subheader("Property Summary")

            summary = pd.DataFrame({

                "Property Detail": [
                    "Location",
                    "Property Type",
                    "BHK",
                    "Size",
                    "Current Price",
                    "Price per SqFt",
                    "Property Age",
                    "Infrastructure Score"
                ],

                "Value": [
                    f"{locality}, {city}, {state}",
                    property_type,
                    bhk,
                    f"{size:,.0f} SqFt",
                    f"₹ {price:.2f} Lakhs",
                    f"₹ {exact_price_per_sqft * 100000:,.0f}",
                    f"{age:,.0f} years",
                    f"{infrastructure_score:.3f}"
                ]

            })

            st.table(summary)

            # ==========================================
            # DISCLAIMER
            # ==========================================

            st.info(
                "Important: The five-year price estimate is based on "
                "the project's simulated 8% annual appreciation target. "
                "It is not a verified market forecast. The investment "
                "classification is based on rule-derived training labels "
                "and should not be treated as financial advice. "
                "Classification probabilities are model estimates, "
                "not guarantees of actual investment performance."
            )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )


# ==================================================
# 8. MARKET INSIGHTS
# ==================================================

elif selected_page == "Market Insights":

    st.title("📊 Market Insights")
    st.write(
        "Explore descriptive patterns in the supplied housing dataset. "
        "These charts describe the dataset and do not establish causation."
    )
    st.divider()

    f1, f2 = st.columns(2)

    with f1:
        selected_state = st.selectbox(
            "Filter by State",
            ["All States"] + sorted(df["State"].dropna().unique().tolist()),
            key="market_state"
        )

    with f2:
        property_types = ["All Property Types"] + sorted(
            df["Property_Type"].dropna().unique().tolist()
        )
        selected_type = st.selectbox(
            "Filter by Property Type",
            property_types,
            key="market_property_type"
        )

    market_df = df.copy()

    if selected_state != "All States":
        market_df = market_df[market_df["State"] == selected_state]

    if selected_type != "All Property Types":
        market_df = market_df[market_df["Property_Type"] == selected_type]

    if market_df.empty:
        st.warning("No records match the selected filters.")
    else:
        a, b, c = st.columns(3)
        a.metric("Filtered Properties", f"{len(market_df):,}")
        b.metric("Average Price", f"₹{market_df['Price_in_Lakhs'].mean():.2f} L")
        c.metric("Median Price", f"₹{market_df['Price_in_Lakhs'].median():.2f} L")

        st.divider()

        chart_left, chart_right = st.columns(2)

        with chart_left:
            st.subheader("Average Price by City")
            city_avg = (
                market_df.groupby("City")["Price_in_Lakhs"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .sort_values()
            )
            st.bar_chart(city_avg, horizontal=True)

        with chart_right:
            st.subheader("Average Price by Property Type")
            type_avg = (
                market_df.groupby("Property_Type")["Price_in_Lakhs"]
                .mean()
                .sort_values(ascending=False)
            )
            st.bar_chart(type_avg)

        st.divider()
        st.subheader("Average Price by BHK")
        bhk_avg = (
            market_df.groupby("BHK")["Price_in_Lakhs"]
            .mean()
            .sort_index()
        )
        st.line_chart(bhk_avg)

        st.subheader("Price Distribution")
        st.bar_chart(
            market_df["Price_in_Lakhs"]
            .round(-1)
            .value_counts()
            .sort_index()
            .head(30)
        )

        st.caption(
            "Interpretation: these are descriptive statistics from the provided "
            "dataset. They should not be interpreted as live property-market prices."
        )


# ==================================================
# 9. MODEL PERFORMANCE
# ==================================================

elif selected_page == "Model Performance":

    st.title("🤖 Model Performance")
    st.write("Evaluation results recorded during the project's model experiments.")
    st.divider()

    st.subheader("Classification — Updated Logistic Regression")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", "88.74%")
    m2.metric("Precision", "89.13%")
    m3.metric("Recall", "89.50%")
    m4.metric("F1-score", "89.32%")

    st.caption("Confusion matrix: rows = actual class [0, 1], columns = predicted class [0, 1].")
    confusion = pd.DataFrame(
        [[20841, 2869], [2760, 23530]],
        index=["Actual: Not Good", "Actual: Good"],
        columns=["Predicted: Not Good", "Predicted: Good"]
    )
    st.dataframe(confusion, use_container_width=True)

    st.divider()
    st.subheader("Regression — Linear Regression")

    r1, r2, r3 = st.columns(3)
    r1.metric("RMSE", "0.000868")
    r2.metric("MAE", "0.000635")
    r3.metric("R²", "1.000000")

    st.warning(
        "The regression target was synthetically generated using an 8% fixed "
        "annual appreciation formula, and current price is an input feature. "
        "Near-perfect regression metrics therefore demonstrate reproduction "
        "of that formula, not real-world forecasting ability."
    )

    st.info(
        "The classification target is also rule-derived. These metrics measure "
        "agreement with the project's generated labels, not independently "
        "verified investment profitability."
    )


# ==================================================
# 10. ABOUT PROJECT
# ==================================================

elif selected_page == "About Project":

    st.title("ℹ️ About the Project")
    st.subheader("Real Estate Investment Advisor")
    st.write(
        "An AI/ML project that classifies property records using a "
        "project-defined investment label and estimates a simulated "
        "five-year property value."
    )

    st.divider()
    st.subheader("Project Objectives")
    st.markdown(
        """
        - Classify whether a property is a good investment according to
          the project's rule-derived target.
        - Estimate a five-year property value using the simulated
          8% annual appreciation assumption.
        - Explore descriptive patterns in the supplied housing dataset.
        - Track model experiments with MLflow.
        """
    )

    st.subheader("Technology Stack")
    st.markdown(
        """
        - **Programming:** Python
        - **Data processing:** Pandas, NumPy
        - **Machine learning:** Scikit-learn, XGBoost
        - **Experiment tracking:** MLflow
        - **Application:** Streamlit
        - **Visualization:** Streamlit charts and data visualizations
        """
    )

    st.subheader("Dataset")
    st.write(f"Records available: **{len(df):,}**")
    st.write(f"Dataset columns: **{len(df.columns)}**")

    st.subheader("Important Limitations")
    st.markdown(
        """
        1. The classification label is generated from predefined dataset rules;
           it is not verified investment performance.
        2. The future-price target assumes fixed 8% annual appreciation.
        3. The dataset is not a live property listing or real-time market feed.
        4. Predictions are educational project outputs, not financial advice.
        """
    )

    st.divider()
    st.caption("Real Estate Investment Advisor | Machine Learning Project | Version 1.0")


