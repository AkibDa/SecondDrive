import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

