import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------------------------------------------------------------------------------------------------

grocery_to_import_map = {
    'Alcohol': 'Beverages',
    'Beverages': 'Beverages',
    'Commercially prepared items': 'Other',
    'Dairy': 'Dairy',
    'Fats and oils': 'VegetablesOil',
    'Fruits': 'Fruits',
    'Grains': 'Grains',
    'Meats, eggs, and nuts': 'Meats',  # or you can aggregate Meats+Animals+Nuts
    'Other': 'Other',
    'Sugar and sweeteners': 'Sweets',
    'Vegetables': 'Vegetables'
}

avg_tariff_to_years = {
    2000: 0.0210,
    2001: 0.0211,
    2002: 0.0216,
    2003: 0.0196,
    2004: 0.0179,
    2005: 0.0175,
    2006: 0.0170,
    2007: 0.0155,
    2008: 0.0158,
    2009: 0.0171,
    2010: 0.0166,
    2011: 0.0167,
    2012: 0.0167,
    2013: 0.0167,
    2014: 0.0169,
    2015: 0.0169,
    2016: 0.0165,
    2017: 0.0166,
    2018: 0.0159,
    2019: 0.0158,
    2020: 0.0152,
    2021: 0.0148,
    2022: 0.0154,
    2023: 0.0150,
    2024: 0.0250,
}

def convert_to_dollars(row):
    if row['UOM'] == 'Million $':
        return row['FoodValue'] * 1_000_000
    elif row['UOM'] in ['1,000', '1,000 mt', '1,000 litpf']:
        return row['FoodValue'] * 1_000
    elif row['UOM'] in ['Dollars', 'Dollars per mt', 'Dollars per KL']:
        return row['FoodValue']
    else:  # percent or unknown
        return np.nan


# ----------------------------------------------------------------------------------------------------------------------

path = kagglehub.dataset_download("danielcalvoglez/us-tariffs-2025")
tariff_df = pd.read_csv(path + "/Tariff Calculations.csv", sep=";", on_bad_lines="skip")

path = kagglehub.dataset_download("pratyushpuri/grocery-store-sales-dataset-in-2025-1900-record")
grocery_df = pd.read_csv(
    "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/100189/StateAndCategory.csv?v=37188",
    encoding="cp1252"
)

imports_df = pd.read_csv(
    "https://ers.usda.gov/sites/default/files/_laserfiche/DataFiles/53736/FoodImports.csv",
    encoding="cp1252"
)

# ----------------------------------------------------------------------------------------------------------------------

grocery_df['Date'] = pd.to_datetime(grocery_df['Date'])
grocery_df['YearMonth'] = grocery_df['Date'].dt.to_period('M').dt.to_timestamp()
#grocery_df = grocery_df[grocery_df['State'] == 'Pennsylvania']
grocery_df['Category'] = grocery_df['Category'].map(grocery_to_import_map)

grocery_df = grocery_df.pivot_table(
    index=['YearMonth', 'Category'],
    columns='Variable',
    values='Value',
    aggfunc='sum'
).reset_index()


imports_df['FoodValue'] = imports_df.apply(convert_to_dollars, axis=1)
imports_df['YearNum'] = pd.to_numeric(imports_df['YearNum'], errors="coerce")
imports_df = imports_df.dropna(subset=["YearNum"])
imports_df['Year'] = imports_df['YearNum'].astype(int)

imports_df['Country'] = imports_df['Country'].str.lower().str.strip()

drop_countries = ['world', 'world (quantity)']

imports_df = imports_df[~imports_df['Country'].str.lower().isin(drop_countries)]
imports_df['Share'] = (
    imports_df
    .groupby(['Category','Year'])['FoodValue']
    .transform(lambda x: x / x.sum())
)

tariff_df['Country'] = (
    tariff_df['Country']
        .str.lower()
        .str.replace('**', '')
        .str.replace('*', '')
        .str.strip()
)

# ----------------------------------------------------------------------------------------------------------------------

grocery_agg = (
    grocery_df
    .groupby(['Category', 'YearMonth'])
    .agg(
        total_sales=('Dollars', 'sum'),
        total_units=('Unit sales', 'sum')
    ).reset_index()
)

grocery_agg['avg_price'] = grocery_agg['total_sales'] / grocery_agg['total_units']

print(grocery_agg)

# ----------------------------------------------------------------------------------------------------------------------

# Repeat each row 12 times for months
imports_monthly = imports_df.loc[imports_df.index.repeat(12)].copy()
imports_monthly['Month'] = list(range(1,13)) * len(imports_df)
imports_monthly['YearMonth'] = pd.to_datetime(imports_monthly['Year'].astype(str) + '-' + imports_monthly['Month'].astype(str) + '-01')

imports_agg = (
    imports_monthly
    .groupby(['Country', 'Category', 'YearMonth', 'Share'])
    .agg(
        import_price_mean=('FoodValue', 'mean'),
        import_value_total=('FoodValue', 'sum'),
    )
    .reset_index()
)

print('Import Categories:', imports_agg['Category'].unique())
print('Grocery Categories:', grocery_agg['Category'].unique())


# ----------------------------------------------------------------------------------------------------------------------

merged = pd.merge(
    grocery_agg,
    imports_agg,
    left_on=['Category', 'YearMonth'],
    right_on = ['Category', 'YearMonth'],
    how='left'
)

historical = merged[merged['YearMonth'] < '2024-01']

# ----------------------------------------------------------------------------------------------------------------------

tariff_agg = (
    tariff_df
    .groupby('Country')
    .agg(
        Tariff=('Trump Tariffs Alleged','max')
    )
    .reset_index()
)

tariff_agg['Tariff'] = (
        tariff_agg['Tariff']
        .str.rstrip('%')
        .astype(float) / 100
)

merged_with_tariff = pd.merge(
    merged,
    tariff_agg[['Country','Tariff']],
    on='Country',
    how='left'
)

predict_df = merged_with_tariff.groupby(['Category','YearMonth']).agg(
    avg_price=('avg_price', 'mean')
).reset_index()

future_2024 = []

for cat in predict_df['Category'].unique():
    subset = predict_df[predict_df['Category'] == cat].sort_values('YearMonth')
    x = np.arange(len(subset))
    period = 19
    y = subset['avg_price'].values

    slope, intercept = np.polyfit(x, y, 1)
    last_date = subset['YearMonth'].max()
    dates_2024 = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods = period,
        freq='MS'
    )
    x_future = np.arange(len(subset), len(subset) + period)
    future_prices = slope * x_future + intercept

    df_future = pd.DataFrame({
        'Category': cat,
        'YearMonth': dates_2024,
        'avg_price': future_prices
    })

    future_2024.append(df_future)

category_tariff_share = (
    merged_with_tariff.groupby('Category')[['Share', 'Tariff']]
    .mean()
    .reset_index()
)



future_2024_df = pd.concat(future_2024, ignore_index=True)

future_2024_df = pd.merge(
    future_2024_df,
    category_tariff_share,
    on='Category',
    how='left'
)

print(future_2024_df.head())

country_stats = merged_with_tariff.groupby(['Category','Country'])[['Share','Tariff']].mean().reset_index()

future_2024_country = future_2024_df.merge(
    country_stats,
    on='Category',
    how='left'
)

future_2024_country = future_2024_country.rename(
    columns={'Share_y':'Share','Tariff_y':'Tariff'}
)[['Category','YearMonth','avg_price','Country','Share','Tariff']]

merged_with_tariff = pd.concat([
    merged_with_tariff,
    future_2024_country
], ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------

actual_2025 = merged_with_tariff[
    (merged_with_tariff['YearMonth'].dt.year == 2025) & (merged_with_tariff['avg_price'].notna())
]

future_2025 = merged_with_tariff[merged_with_tariff['YearMonth'].dt.year == 2024].copy()
future_2025['YearMonth'] = future_2025['YearMonth'] + pd.DateOffset(years=1)
future_2025['predicted_2025'] = (future_2025['avg_price'] * (1 + future_2025['Tariff']))
future_2025['avg_price'] = future_2025['predicted_2025']
future_2025 = future_2025.drop(columns=['predicted_2025'])
future_2025 = future_2025[['Category','YearMonth','avg_price','Share','Tariff','Country']]

merged_with_tariff = pd.concat([
    merged_with_tariff[merged_with_tariff['YearMonth'].dt.year < 2025],
    actual_2025,
    future_2025
], ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------

sp500 = yf.download("^GSPC", start="2018-01-01", interval="1mo", auto_adjust=False)

sp500 = sp500.reset_index()

# Flatten columns if MultiIndex exists
sp500.columns = [col[0] if isinstance(col, tuple) else col for col in sp500.columns]

# Rename columns
sp500 = sp500.rename(columns={'Date': 'YearMonth', 'Close': 'SP500'})

# Convert to timestamp
sp500['YearMonth'] = sp500['YearMonth'].dt.to_period('M').dt.to_timestamp()


merged_with_tariff = pd.merge(
    merged_with_tariff,
    sp500[['YearMonth', 'SP500']],
    on='YearMonth',
    how='left'
)

# ----------------------------------------------------------------------------------------------------------------------

final_ml_df = merged_with_tariff.dropna(subset=['avg_price'])

tariff_map = {year: val for year, val in avg_tariff_to_years.items() if year != 2025}

years = final_ml_df['YearMonth'].dt.year
final_ml_df.loc[years != 2025, 'Tariff'] = final_ml_df['YearMonth'].dt.year.map(tariff_map).fillna(0)


final_ml_df.to_csv("tariff_price_data.csv", index=False)

print(final_ml_df['Country'].unique())

plot_df = final_ml_df.groupby(['Category','YearMonth']).agg(
    avg_price=('avg_price', 'mean')
).reset_index()

print("Final ML dataset:")
print(final_ml_df.head())
print("Columns available for prediction:")
print(final_ml_df.columns)

plt.figure(figsize=(12, 6))

for cat in plot_df['Category'].unique():
    subset = plot_df[plot_df['Category'] == cat]
    subset = subset.sort_values('YearMonth')
    line, = plt.plot(subset['YearMonth'], subset['avg_price'], marker='o', label=cat)
    color = line.get_color()

    x = np.arange(len(subset))  # simple numeric x-axis
    y = subset['avg_price'].values
    slope, intercept = np.polyfit(x, y, 1)

    extended_months = 32
    x_extended = np.arange(len(subset) + extended_months)
    trend_extended = slope * x_extended + intercept

    extended_dates = pd.date_range(start=subset['YearMonth'].min(), periods=len(subset) + extended_months, freq='MS')

    plt.plot(extended_dates, trend_extended, linestyle='--', color=color)

plt.axvline(pd.Timestamp('2025-08-31'), color='red', linestyle='--', label='Start of Predicted 2025')
plt.title(f"Average Price Over Time by USDA Category for PA")
plt.xlabel("Month")
plt.ylabel("Average Price ($)")
plt.legend()
plt.grid(True)
plt.show()

country_imports = imports_agg.groupby(['Country', 'YearMonth']).agg(
    monthly_import_value=('import_value_total', 'sum')
).reset_index()

plt.figure(figsize=(14,6))

top_countries = country_imports.groupby('Country')['monthly_import_value'].sum().nlargest(5).index

for country in top_countries:
    subset = country_imports[country_imports['Country'] == country].sort_values('YearMonth')
    plt.plot(subset['YearMonth'], subset['monthly_import_value'], marker='o', label=country.capitalize())

plt.title("Monthly Food Imports by Country")
plt.xlabel("Month")
plt.ylabel("Import Value ($)")
plt.legend()
plt.grid(True)
plt.show()

