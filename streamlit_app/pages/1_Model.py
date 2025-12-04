import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer



# Helper function to simulate tariff effect
def simulate_category_price_effect(model, df, features, country, new_tariff, from_year=None):
    """
    Simulate what happens to category prices if a single country's tariff is changed.
    - model: trained sklearn Pipeline
    - df:    DataFrame with at least [Country, Category, Share, Tariff, YearMonthNum, SP500]
    - features: list of feature column names used to train the model
    - country: country name (string)
    - new_tariff: float (e.g., 0.10 for 10%)
    - from_year: if not None, only change tariff from this year onward
    """
    df_copy = df.copy()

    # Normalize country name
    df_copy["Country"] = df_copy["Country"].astype(str).str.strip().str.lower()

    # Mask rows for that country (and optional year)
    mask = df_copy["Country"] == country.lower()
    if from_year is not None:
        years = pd.to_datetime(df_copy["YearMonthNum"], unit="s").dt.year
        mask &= (years >= from_year)

    # Apply new tariff
    df_copy.loc[mask, "Tariff"] = new_tariff

    # Fill numeric features that might have NaNs
    for col in ["Share", "Tariff", "YearMonthNum", "SP500"]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)

    # Predict prices
    df_copy["predicted_price"] = model.predict(df_copy[features])

    # Weighted average by category using Share as weights
    def weighted_avg_price(group):
        weights = df_copy.loc[group.index, "Share"]
        return np.sum(group["predicted_price"] * weights) / np.sum(weights)

    category_avg = (
        df_copy
        .groupby("Category")
        .apply(weighted_avg_price)
        .reset_index(name="predicted_category_price")
    )

    return category_avg



# Load data & train model (cached so it only runs once)
@st.cache_resource
def load_model_and_data():
    # Path to CSV (one folder up from streamlit_app/)
    data_path = Path(__file__).resolve().parents[1] / "../tariff_price_data.csv"
    final_ml_df = pd.read_csv(data_path)

    # Recreate the same feature engineering as Model_Training.py
    df = final_ml_df.copy()

    # Ensure datetime and numeric time
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])
    df["YearMonthNum"] = df["YearMonth"].astype("int64") // 10**9  # seconds since epoch

    # Interaction term
    df["Tariff_Share"] = df["Tariff"] * df["Share"]

    # Features (same list as in Model_Training.py)
    features = ["Share", "Tariff", "YearMonthNum", "Category", "Country", "SP500", "Tariff_Share"]

    # Drop columns not used as features or target
    X = df[features]
    y = df["avg_price"]

    # Train/val split (kept for consistency; you could also train on all data)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Numeric and categorical features
    numeric_features = ["Share", "Tariff", "YearMonthNum", "SP500"]
    categorical_features = ["Category", "Country"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ],
        remainder="passthrough",  # Tariff_Share will pass through here
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)

    # Return model + full df used for simulation (X with original non-feature columns attached)
    # We'll keep a simulation DataFrame that has both the features and the
    # extra columns needed for grouping/weights.
    sim_df = X.copy()
    # Attach Share & Category explicitly to be safe
    sim_df["Share"] = df["Share"]
    sim_df["Category"] = df["Category"]
    sim_df["Country"] = df["Country"]
    sim_df["SP500"] = df["SP500"]

    return model, features, sim_df



# Streamlit UI
def main():
    st.title("🧮 Tariff Impact Model")
    st.markdown(
        "This page uses the trained regression model to estimate how changing "
        "tariffs for a specific country might affect **average grocery prices by category**."
    )

    # Load model & data
    with st.spinner("Loading model and data..."):
        model, features, sim_df = load_model_and_data()

    # Sidebar inputs
    st.sidebar.header("Simulation Settings")

    # Available countries from data
    available_countries = (
        sim_df["Country"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", np.nan)
        .dropna()
        .unique()
    )
    available_countries = np.sort(available_countries)

    country = st.sidebar.selectbox(
        "Country to adjust tariff for:",
        options=available_countries,
        index=0,
        format_func=lambda c: c.capitalize(),
    )

    tariff_percent = st.sidebar.slider(
        "New Tariff Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5,
        help="Enter the tariff level you want to simulate (e.g., 10% = 0.10).",
    )

    use_from_year = st.sidebar.checkbox(
        "Apply tariff only from a given year onward?", value=False
    )
    from_year = None
    if use_from_year:
        # Derive year range from data
        years = pd.to_datetime(sim_df["YearMonthNum"], unit="s").dt.year
        min_year = int(years.min())
        max_year = int(years.max())
        from_year = st.sidebar.slider(
            "Apply new tariff starting from year:",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
        )


    st.markdown("### 🔍 Current baseline")
    st.write(
        "The model is trained on historical data from `tariff_price_data.csv`. "
        "Below is a quick peek at the features used for prediction:"
    )
    st.dataframe(sim_df.head())

    # Run simulation when user clicks button
    if st.button("Run Tariff Simulation", type="primary"):
        new_tariff_fraction = tariff_percent / 100.0

        with st.spinner("Simulating new tariff scenario..."):
            result_df = simulate_category_price_effect(
                model=model,
                df=sim_df,
                features=features,
                country=country,
                new_tariff=new_tariff_fraction,
                from_year=from_year,
            )

        st.markdown(
            f"### 📈 Predicted Category Prices for `{country.capitalize()}` at **{tariff_percent:.2f}%** tariff"
            + (f" (from {from_year} onward)" if from_year is not None else "")
        )

        st.dataframe(result_df.style.format({"predicted_category_price": "{:.4f}"}))

        # Simple bar chart
        st.bar_chart(
            data=result_df.set_index("Category")["predicted_category_price"],
            use_container_width=True,
        )

        st.caption(
            "Prices are **model predictions**, not actual observed values. "
            "Only the selected country's tariff was changed; all other countries "
            "remain at their historical tariff levels."
        )
    else:
        st.info("Set a country and tariff, then click **Run Tariff Simulation**.")


if __name__ == "__main__":
    main()
