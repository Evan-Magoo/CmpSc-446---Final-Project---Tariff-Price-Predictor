import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


def simulate_category_price_effect(model, df, features, country, new_tariff, from_year=None):
    df_copy = df.copy()
    df_copy['Country'] = df_copy['Country'].str.strip().str.lower()

    # Apply mask
    mask = df_copy['Country'] == country.lower()
    if from_year is not None:
        mask &= (pd.to_datetime(df_copy['YearMonthNum'], unit='s').dt.year >= from_year)

    df_copy.loc[mask, 'Tariff'] = new_tariff

    # Fill numeric features
    for col in ['Share', 'Tariff', 'YearMonthNum', 'SP500']:
        df_copy[col] = df_copy[col].fillna(0)

    # Predict
    df_copy['predicted_price'] = model.predict(df_copy[features])

    # Weighted category average (fix FutureWarning)
    category_avg = df_copy.groupby('Category').agg(
        predicted_category_price=('predicted_price', lambda x: np.sum(x * df_copy.loc[x.index, 'Share']) / np.sum(df_copy.loc[x.index, 'Share']))
    ).reset_index()

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
    final_ml_df = pd.read_csv('tariff_price_data.csv')


    features = ['Share', 'Tariff', 'YearMonthNum', 'Category', 'Country', 'SP500']
    df = final_ml_df.copy()
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])
    df['YearMonthNum'] = df['YearMonth'].astype('int64') // 10**9
    df['Tariff_Share'] = df['Tariff'] * df['Share']
    features.append('Tariff_Share')
    df = df.drop(columns=['YearMonth'])

    X = df[features]
    y = df['avg_price']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

    numeric_features = ['Share', 'Tariff', 'YearMonthNum', 'SP500']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category', 'Country']),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())  # <-- scale numeric features
            ]), numeric_features)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)

    pred_val = model.predict(X_val)

    mae = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))
    r2 = r2_score(y_val, pred_val)

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R² Score:", r2)

    test = X_val.copy()
    test['avg_price'] = y_val

    for tariff in [0, 0.1, 0.25, 0.5 , 1, 10]:
        category_effect = simulate_category_price_effect(model, test, features, 'china', tariff)
        print(f'\nChina Tariff = {tariff * 100}%')
        print(category_effect)

    print()
    print('-' * 100)

    for tariff in [0, 0.1, 0.25, 0.5 , 1, 10]:
        category_effect = simulate_category_price_effect(model, test, features, 'mexico', tariff)
        print(f'\nMexico Tariff = {tariff * 100}%')
        print(category_effect)

    print()
    print('-' * 100)

    for tariff in [0, 0.1, 0.25, 0.5, 1, 10]:
        category_effect = simulate_category_price_effect(model, test, features, 'brazil', tariff)
        print(f'\nBrazil Tariff = {tariff * 100}%')
        print(category_effect)