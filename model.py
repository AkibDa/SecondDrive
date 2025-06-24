import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('used_car_price_dataset_extended.csv')

# Check how many missing values in each column
print(df.isnull().sum())

# Drop rows with too many missing values (optional)
df = df.dropna(thresh=5)  # keeps rows with at least 5 non-NaN values

# Fill missing numerical values with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_cols = ['mileage_kmpl', 'engine_cc', 'make_year']  # example numeric columns

df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

from sklearn.model_selection import train_test_split

columns_to_drop = ['price_usd', 'brand', 'transmission', 'color', 'insurance_valid']
existing_cols = [col for col in columns_to_drop if col in df.columns]

X = df.drop(existing_cols, axis=1)
y = df['price_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred):
  print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
  print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
  print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

model_1 = LinearRegression()
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

model_2 = RandomForestRegressor(random_state=42)
model_2.fit(X_train, y_train)
y_pred_2 = model_2.predict(X_test)

evaluate_model(y_test, y_pred_1)
print(model_1.score(X_test, y_test))
evaluate_model(y_test, y_pred_2)
print(model_2.score(X_test, y_test))

coefficients = pd.Series(model_1.coef_, index=X.columns)
coefficients = coefficients.sort_values(key=abs, ascending=False)
print(coefficients)

# Recreate cleaned X and drop weak features
features_to_drop = [
    'brand_Volkswagen',
    'transmission_Manual',
    'color_White',
    'fuel_type_Petrol',
    'accidents_reported',
    'service_history_Partial'
]

X_cleaned = X.drop(columns=features_to_drop)

# Now split X_cleaned and y to get matching shapes
X_train_cleaned, X_test_cleaned, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.2, random_state=42
)

# Fit the model on the cleaned, properly-split data

model_clean = LinearRegression()
model_clean.fit(X_train_cleaned, y_train)

# Predict and evaluate
y_pred_clean = model_clean.predict(X_test_cleaned)

evaluate_model(y_test, y_pred_clean)
print(model_clean.score(X_test_cleaned, y_test))