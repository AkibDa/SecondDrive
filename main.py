import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load data and models (cached for performance)
@st.cache_data
def load_data():
  df = pd.read_csv('used_car_price_dataset_extended.csv')
  model = joblib.load('car_price_model_accurate.pkl')
  scaler = joblib.load('scaler.pkl')
  feature_names = joblib.load('feature_names.pkl')
  return df, model, scaler, feature_names


def get_available_categories(model, prefix):
  return sorted([f[len(prefix):].lower() for f in model.feature_names_in_ if f.startswith(prefix)])


def safe_category_selectbox(name, prefix, model):
  available = get_available_categories(model, prefix)
  if available:
    return st.selectbox(f"Select {name}", available)
  else:
    st.warning(f"⚠️ {name} not used in this model — selection will be ignored.")
    return None


def check_price_tab(df, model, scaler, feature_names):
  st.header("🚗 Car Price Estimator")

  # Get available brands
  available_brands = get_available_categories(model, 'brand_')
  brand = st.selectbox("Brand", available_brands)

  # Create columns for better layout
  col1, col2 = st.columns(2)

  with col1:
    year = st.number_input("Make Year", min_value=1900, max_value=2025, value=2015)
    cc = st.number_input("Engine CC", min_value=500, max_value=10000, value=1500)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0, max_value=100.0, value=15.0)

  with col2:
    owners = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1)
    accidents = st.number_input("Number of Accidents Reported", min_value=0, max_value=10, value=0)
    insurance = st.radio("Is Insurance Valid?", ["Yes", "No"], index=0)

  # Safe category inputs
  transmission = safe_category_selectbox('transmission', 'transmission_', model)
  fuel_type = safe_category_selectbox('fuel type', 'fuel_type_', model)
  color = safe_category_selectbox('color', 'color_', model)
  service = safe_category_selectbox('service history', 'service_history_', model)

  if st.button("Estimate Price"):
    input_dict = {feature: 0 for feature in feature_names}

    # Engineered features
    car_age = 2025 - year
    input_dict['car_age'] = car_age
    input_dict['engine_cc'] = cc
    input_dict['mileage_kmpl'] = mileage
    input_dict['owner_count'] = owners
    input_dict['accidents_reported'] = accidents if 'accidents_reported' in feature_names else 0

    if 'engine_per_age' in feature_names:
      input_dict['engine_per_age'] = cc / (car_age + 1)
    if 'mileage_per_owner' in feature_names:
      input_dict['mileage_per_owner'] = mileage / (owners + 1)

    # One-hot encode
    for feature in feature_names:
      if feature == f'brand_{brand}':
        input_dict[feature] = 1
      elif fuel_type and feature == f'fuel_type_{fuel_type}':
        input_dict[feature] = 1
      elif transmission and feature == f'transmission_{transmission}':
        input_dict[feature] = 1
      elif color and feature == f'color_{color}':
        input_dict[feature] = 1
      elif service and feature == f'service_history_{service}':
        input_dict[feature] = 1
      elif feature == 'insurance_valid_yes':
        input_dict[feature] = 1 if insurance.lower() == "yes" else 0

    # DataFrame & scale numeric columns
    input_df = pd.DataFrame([[input_dict[f] for f in feature_names]], columns=feature_names)
    numeric_cols = ['mileage_kmpl', 'engine_cc', 'car_age', 'engine_per_age', 'mileage_per_owner']
    scaled = [col for col in numeric_cols if col in input_df.columns]
    input_df[scaled] = scaler.transform(input_df[scaled])

    # Predict and inverse log transform
    log_price = model.predict(input_df)[0]
    price = np.expm1(log_price)

    st.success(f"💰 Estimated Resale Price: ${price:,.2f}")

def suggest_car_tab(df):
  st.header("Car Recommendation System")

  budget = st.number_input("Your maximum budget (USD)", min_value=0.0, max_value=1000000.0, value=20000.0)

  col1, col2 = st.columns(2)

  with col1:
    fuel_type = st.selectbox(
      "Preferred fuel type",
      ["Any", "Petrol", "Diesel", "Electric"],
      index=0
    )
    transmission = st.selectbox(
      "Preferred transmission",
      ["Any", "Manual", "Automatic"],
      index=0
    )

  with col2:
    brand = st.text_input("Preferred brand (optional)", "")
    min_mileage = st.number_input("Minimum mileage (kmpl)", min_value=0.0, max_value=100.0, value=0.0)
    min_year = st.number_input("Earliest make year", min_value=1900, max_value=2023, value=2000)

  sort_by = st.radio("Sort results by", ["Mileage", "Price"], index=0)

  if st.button("Find Recommendations"):
    df_filtered = df[df['price_usd'] <= budget]

    if fuel_type != "Any":
      df_filtered = df_filtered[df_filtered['fuel_type'].str.lower() == fuel_type.lower()]
    if transmission != "Any":
      df_filtered = df_filtered[df_filtered['transmission'].str.lower() == transmission.lower()]
    if brand:
      df_filtered = df_filtered[df_filtered['brand'].str.lower() == brand.lower()]

    df_filtered = df_filtered[df_filtered['mileage_kmpl'] >= min_mileage]
    df_filtered = df_filtered[df_filtered['make_year'] >= min_year]

    sort_column = 'price_usd' if sort_by == "Price" else 'mileage_kmpl'
    df_filtered = df_filtered.sort_values(by=sort_column, ascending=(sort_by == "Price"))
    recommended = df_filtered.head(5)

    if recommended.empty:
      st.warning("No cars found matching your criteria. Try increasing your budget or relaxing filters.")
    else:
      st.subheader("Top Car Recommendations")
      st.dataframe(recommended[['brand', 'make_year', 'engine_cc', 'mileage_kmpl', 'price_usd']])


def main():
  st.set_page_config(page_title="Car Price Tool", layout="wide")
  st.title("🚗 Used Car Price Estimator & Recommender")

  # Load data and models
  df, model, scaler, feature_names = load_data()

  # Create tabs
  tab1, tab2 = st.tabs(["Price Estimator", "Car Recommender"])

  with tab1:
    check_price_tab(df, model, scaler, feature_names)

  with tab2:
    suggest_car_tab(df)


if __name__ == '__main__':
  main()