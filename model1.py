import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 🚗 1. Load and Clean Data
df = pd.read_csv('used_car_price_dataset_extended.csv')

# Drop rows with too many missing values
df.dropna(thresh=5, inplace=True)

# Fill numerical with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill categorical with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 🛠️ 2. Feature Engineering
df = df[df['price_usd'] < df['price_usd'].quantile(0.99)]  # Remove top 1% outliers

# Add useful derived features
df['car_age'] = 2025 - df['make_year']
df['engine_per_age'] = df['engine_cc'] / df['car_age'].replace(0, 1)
df['mileage_per_owner'] = df['mileage_kmpl'] / (df['owner_count'] + 1)

# Log-transform the target
df['log_price'] = np.log1p(df['price_usd'])

# Drop original year column
df.drop(columns=['make_year'], inplace=True)

# One-hot encode categoricals
df = pd.get_dummies(df, drop_first=True)

# 🪜 3. Prepare Features and Scaling
target = 'log_price'
scaled_cols = ['mileage_kmpl', 'engine_cc', 'car_age', 'engine_per_age', 'mileage_per_owner']

scaler = StandardScaler()
df[scaled_cols] = scaler.fit_transform(df[scaled_cols])

X = df.drop(columns=['price_usd', 'log_price'])
y = df['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🌲 4. Train and Tune RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# 📊 5. Evaluate with Real-World Metrics
def evaluate_model_log(y_true_log, y_pred_log):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    print(f"MAE: {mean_absolute_error(y_true, y_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

y_pred_log = best_model.predict(X_test)
evaluate_model_log(y_test, y_pred_log)

# Show top features
importances = pd.Series(best_model.feature_importances_, index=X.columns)
print("\nTop Features:")
print(importances.sort_values(ascending=False).head(10))

# 💾 6. Save Model and Scaler
joblib.dump(best_model, 'car_price_model_accurate.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
