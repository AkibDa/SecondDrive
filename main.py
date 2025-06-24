import pandas as pd
import joblib

def get_available_categories(model, prefix):
    return sorted([f[len(prefix):].lower() for f in model.feature_names_in_ if f.startswith(prefix)])

def safe_category_input(name, prefix, model):
    available = get_available_categories(model, prefix)
    if available:
        print(f"\n🟢 Supported {name}s: {', '.join(available)}")
        val = input(f"Enter {name}: ").strip().lower()
        if val not in available:
            print(f"❌ '{val}' is not a valid {name}.")
            return None, available
        return val, available
    else:
        val = input(f"Enter {name} (input will be ignored): ").strip().lower()
        print(f"⚠️ '{name}' not used in this model — input will be ignored.")
        return val, []

def check_price(df, model, scaler):
    # --- Get known brands from model ---
    available_brands = get_available_categories(model, 'brand_')
    print("\n🟢 Supported brands:", ', '.join(available_brands))
    brand = input("Enter brand: ").strip().lower()
    if brand not in available_brands:
        print(f"\n❌ '{brand}' not in supported brands.")
        return

    # --- Get numerical inputs ---
    try:
        year = int(input("Enter year: "))
        cc = float(input("Enter engine cc: "))
        mileage = float(input("Enter mileage (kmpl): "))
        owners = int(input("Enter number of previous owners: "))
        accidents = int(input("Enter number of accidents reported: "))
    except ValueError:
        print("❌ Invalid number. Please enter numeric values correctly.")
        return

    # --- Safe category inputs ---
    transmission, _ = safe_category_input('transmission', 'transmission_', model)
    if transmission is None: return

    fuel_type, _ = safe_category_input('fuel type', 'fuel_type_', model)
    if fuel_type is None: return

    color, _ = safe_category_input('color', 'color_', model)
    if color is None: return

    service, _ = safe_category_input('service history', 'service_history_', model)
    if service is None: return

    insurance = input("Is insurance valid? (Yes/No): ").strip().lower()
    if insurance not in ['yes', 'no']:
        print(f"\n❌ Insurance must be 'Yes' or 'No'")
        return

    # --- Build model input dict ---
    expected_features = model.feature_names_in_
    input_dict = {feature: 0 for feature in expected_features}

    # Fill numeric fields
    input_dict['make_year'] = year
    input_dict['engine_cc'] = cc
    input_dict['mileage_kmpl'] = mileage
    input_dict['owner_count'] = owners
    if 'accidents_reported' in expected_features:
        input_dict['accidents_reported'] = accidents

    # One-hot fields only if model used them
    for feature in expected_features:
        if feature == f'brand_{brand}':
            input_dict[feature] = 1
        elif feature == f'fuel_type_{fuel_type}':
            input_dict[feature] = 1
        elif feature == f'transmission_{transmission}':
            input_dict[feature] = 1
        elif feature == f'color_{color}':
            input_dict[feature] = 1
        elif feature == f'service_history_{service}':
            input_dict[feature] = 1
        elif feature == 'insurance_valid_Yes':
            input_dict[feature] = 1 if insurance == 'yes' else 0

    # --- Create input DataFrame ---
    input_df = pd.DataFrame([[input_dict[f] for f in expected_features]], columns=expected_features)

    # --- Scale numeric columns ---
    numeric_to_scale = ['mileage_kmpl', 'engine_cc', 'car_age'] if 'car_age' in expected_features else ['mileage_kmpl', 'engine_cc', 'make_year']
    input_df[numeric_to_scale] = scaler.transform(input_df[numeric_to_scale])

    # --- Predict price ---
    predicted_price = model.predict(input_df)[0]
    print(f"\n💰 Estimated Resale Price: ${predicted_price:,.2f}")

def suggest_car(df):
    budget = float(input("Enter your maximum budget in USD: "))
    fuel_type = input("Enter preferred fuel type (Petrol/Diesel/Electric or leave blank): ").strip().lower()
    transmission = input("Preferred transmission (Manual/Automatic or leave blank): ").strip().lower()
    brand = input("Preferred brand (or leave blank): ").strip().lower()
    min_mileage = input("Minimum mileage (kmpl, optional): ").strip()
    min_year = input("Earliest make year (optional): ").strip()

    df_filtered = df[df['price_usd'] <= budget]

    if fuel_type:
        df_filtered = df_filtered[df_filtered['fuel_type'].str.lower() == fuel_type]
    if transmission:
        df_filtered = df_filtered[df_filtered['transmission'].str.lower() == transmission]
    if brand:
        df_filtered = df_filtered[df_filtered['brand'].str.lower() == brand]
    if min_mileage:
        df_filtered = df_filtered[df_filtered['mileage_kmpl'] >= float(min_mileage)]
    if min_year:
        df_filtered = df_filtered[df_filtered['make_year'] >= int(min_year)]

    sort_by = input("Sort results by 'price' or 'mileage'? (default: mileage): ").strip().lower()
    if sort_by not in ['price', 'mileage']:
        sort_by = 'mileage'

    sort_column = 'price_usd' if sort_by == 'price' else 'mileage_kmpl'
    df_filtered = df_filtered.sort_values(by=sort_column, ascending=(sort_by == 'price')).head(5)
    recommended = df_filtered.sort_values(by=['mileage_kmpl'], ascending=False).head(5)

    if recommended.empty:
        print("\n❌ No cars found matching your criteria. Try increasing your budget or relaxing filters.")
    else:
        print("\n🔎 Top Car Recommendations:\n")
        print(recommended[['brand', 'make_year', 'engine_cc', 'mileage_kmpl', 'price_usd']].to_string(index=False))

    retry = input("Would you like to try again with a higher budget or fewer filters? (yes/no): ").strip().lower()
    if retry == 'yes':
        suggest_car(df)

if __name__ == '__main__':
    df = pd.read_csv('used_car_price_dataset_extended.csv')
    model = joblib.load('car_price_model_accurate.pkl')
    scaler = joblib.load('scaler.pkl')

    choice = int(input('Enter 1 for Check Price Mode, 2 for Suggest Car Mode or anything else for exciting: '))
    if choice == 1:
        check_price(df, model, scaler)
    elif choice == 2:
        suggest_car(df)
    else:
        pass
