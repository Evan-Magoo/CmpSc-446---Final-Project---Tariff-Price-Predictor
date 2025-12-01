import pandas as pd
import numpy as np
import xgboost as xgb
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


def simulate_category_price_effect(model, df, features, country, new_tariff):
    df_copy = df.copy()
    df_copy['Country'] = df_copy['Country'].str.strip().str.lower()

    # Set new tariff for the target country in 2025
    mask = (df_copy['Country'] == country.lower()) & \
           (df_copy['YearMonthNum'] >= pd.Timestamp('2025-01-01').value // 10**9)
    df_copy.loc[mask, 'Tariff'] = new_tariff

    # Fill numeric features
    numeric_features = ['Share', 'Tariff', 'YearMonthNum', 'SP500']
    for col in numeric_features:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)

    # Predict prices
    df_copy['predicted_price'] = model.predict(df_copy[features])

    # Compute category-level weighted average
    category_avg = (
        df_copy
        .groupby('Category', as_index=False)
        .apply(lambda x: pd.Series({'predicted_category_price': np.sum(x['predicted_price'] * x['Share']) / np.sum(x['Share'])}))
        .reset_index(drop=True)
    )

    return category_avg

def simulate_tariff(model, df, features, country_tariff_map):
    df_copy = df.copy()

    # Normalize country column
    if 'Country' in df_copy.columns:
        df_copy['Country'] = df_copy['Country'].str.strip().str.lower()
        for country, tariff in country_tariff_map.items():
            mask = df_copy['Country'] == country.lower()
            df_copy.loc[mask, 'Tariff'] = tariff

    # Fill numeric features
    numeric_features = ['Share', 'Tariff', 'YearMonthNum', 'SP500']
    for col in numeric_features:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)

    # Predict
    df_copy['predicted_price'] = model.predict(df_copy[features])

    return df_copy[['Category', 'Country', 'Tariff', 'predicted_price']]

if __name__ == '__main__':
    final_ml_df = pd.read_csv('../446-Final-Project/tariff_price_data.csv')

    features = ['Share', 'Tariff', 'YearMonthNum', 'Category', 'Country', 'SP500']
    df = final_ml_df.copy()
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    df['YearMonthNum'] = df['YearMonth'].astype('int64') // 10**9
    df = df.drop(columns=['YearMonth'])

    train = df[df['YearMonthNum'] < pd.Timestamp("2025-01-01").value // 10**9]
    test  = df[(df['YearMonthNum'] >= pd.Timestamp("2025-01-01").value // 10**9) &
               (df['YearMonthNum'] < pd.Timestamp("2026-01-01").value // 10**9)]

    X_train = train[features]
    y_train = train['avg_price']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category','Country']),
            ('num', SimpleImputer(strategy='mean'), ['Share', 'Tariff', 'YearMonthNum', 'SP500'])
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    X_test = test[features]
    predicted = model.predict(X_test)
    test = test.copy()
    test.loc[:, 'predicted'] = predicted

    mae = mean_absolute_error(test['avg_price'], test['predicted'])
    rmse = np.sqrt(mean_squared_error(test['avg_price'], test['predicted']))
    r2 = r2_score(test['avg_price'], test['predicted'])

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)

    category_effect = simulate_category_price_effect(model, test, features, 'china', 100)
    print('\nChina Tariff = 1000')
    print(category_effect)


    category_effect = simulate_category_price_effect(model, test, features, 'china', 500)
    print('\nChina Tariff = 500')
    print(category_effect)