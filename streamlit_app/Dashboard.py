import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tariff Price Predictor",
    layout="wide"
)

st.title("📊 Tariff Price Predictor Dashboard")
st.markdown(
    "Explore how food prices, import shares, and tariffs evolve over time, "
    "and how they may be affected by trade policies."
)

# --------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("../tariff_price_data.csv", parse_dates=["YearMonth"])
    # Just in case there are any weird spaces or mixed cases
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip()
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "❌ `tariff_price_data.csv` not found.\n\n"
        "Make sure to run `Final_Project.py` first so the dataset is generated."
    )
    st.stop()

# --------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------
st.sidebar.header("Filters")

categories = sorted(df["Category"].dropna().unique().tolist())
countries = sorted(df["Country"].dropna().unique().tolist())

selected_categories = st.sidebar.multiselect(
    "USDA Category",
    options=categories,
    default=categories  # start with all
)

# Add an "All countries" option
country_options = ["(All countries)"] + countries
selected_countries = st.sidebar.multiselect(
    "Import Country",
    options=country_options,
    default=["(All countries)"]
)

# Date range slider
min_date = df["YearMonth"].min()
max_date = df["YearMonth"].max()

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
)


# Apply filters
mask = (df["YearMonth"] >= pd.Timestamp(date_range[0])) & \
       (df["YearMonth"] <= pd.Timestamp(date_range[1]))

if selected_categories:
    mask &= df["Category"].isin(selected_categories)

# Countries filter
if "(All countries)" not in selected_countries:
    mask &= df["Country"].isin(selected_countries)

filtered = df[mask].copy()

if filtered.empty:
    st.warning("No data for the selected filters. Try broadening the date range or categories.")
    st.stop()

# Aggregate to category-level avg price over time
plot_df = (
    filtered
    .groupby(["Category", "YearMonth"], as_index=False)
    .agg(avg_price=("avg_price", "mean"))
)

# Layout: charts + data
tab1, tab2, tab3 = st.tabs(["📈 Category Prices Over Time", "🌍 Country & Tariffs", "📄 Raw Data"])


# TAB 1: Category price trends
with tab1:
    st.subheader("Average Grocery Price Over Time (by Category)")
    fig, ax = plt.subplots(figsize=(10, 5))

    for cat in sorted(plot_df["Category"].unique()):
        sub = plot_df[plot_df["Category"] == cat].sort_values("YearMonth")
        ax.plot(sub["YearMonth"], sub["avg_price"], marker="o", label=cat)

    ax.set_xlabel("Date")
    ax.set_ylabel("Average Price ($)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown("ℹ️ This chart shows the average price for each selected USDA category over time.")


# TAB 2: Country imports & tariff relationship
with tab2:
    st.subheader("Tariffs vs Price (by Country & Category)")

    # Let user pick one category and one country to zoom in
    col1, col2 = st.columns(2)
    with col1:
        cat_for_country = st.selectbox(
            "Category for country-level view",
            options=selected_categories if selected_categories else categories
        )
    with col2:
        # Countries available for that category within the filtered data
        available_countries = sorted(
            filtered[filtered["Category"] == cat_for_country]["Country"].dropna().unique().tolist()
        )
        if not available_countries:
            st.info(f"No country data available for category `{cat_for_country}` in the current filter.")
        country_for_view = st.selectbox(
            "Country",
            options=available_countries if available_countries else countries
        ) if available_countries else None

    if country_for_view:
        sub = filtered[
            (filtered["Category"] == cat_for_country) &
            (filtered["Country"] == country_for_view)
        ].sort_values("YearMonth")

        if sub.empty:
            st.info("No rows for this category/country in the selected date range.")
        else:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(sub["YearMonth"], sub["avg_price"], marker="o", label="Avg Price ($)")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Average Price ($)")
            ax2.grid(True)

            # If Tariff column exists, add secondary axis
            if "Tariff" in sub.columns:
                ax3 = ax2.twinx()
                ax3.plot(sub["YearMonth"], sub["Tariff"], linestyle="--", marker="x", label="Tariff")
                ax3.set_ylabel("Tariff (fraction)")
            st.pyplot(fig2)

            st.markdown(
                f"Showing **{cat_for_country}** prices for **{country_for_view}**, "
                f"with tariffs over time when available."
            )


# TAB 3: Raw data
with tab3:
    st.subheader("Filtered Dataset")
    st.dataframe(
        filtered.sort_values(["YearMonth", "Category", "Country"])
    )
    st.caption("This is the subset of `tariff_price_data.csv` after applying your filters.")
