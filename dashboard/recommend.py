import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import os
from django.conf import settings

# Load dataset
car_data = pd.read_csv(os.path.join(settings.BASE_DIR, 'car_data.csv'))

def preprocess_data(data):
    df = data.copy()
    df.rename(columns={
        'Make': 'Brand',
        'Engine Fuel Type': 'Fuel_Type',
        'Transmission Type': 'Transmission',
        'Vehicle Size': 'Body_Type',
        'MSRP': 'Price'
    }, inplace=True)

    df.ffill(inplace=True)

    # Clean Fuel Types
    def clean_fuel(fuel):
        fuel = fuel.lower()
        if 'diesel' in fuel:
            return 'DIESEL'
        elif 'electric' in fuel:
            return 'ELECTRIC'
        elif 'hybrid' in fuel:
            return 'HYBRID'
        else:
            return 'PETROL'

    df['Fuel_Type'] = df['Fuel_Type'].apply(clean_fuel)
    df['Brand'] = df['Brand'].str.upper()
    df['Transmission'] = df['Transmission'].str.upper()
    df['Body_Type'] = df['Body_Type'].str.upper()

    return df

def train_kmeans(df, n_clusters=5):
    features = ['Price', 'Brand', 'Fuel_Type', 'Transmission', 'Body_Type', 'Year']
    df_copy = df.copy()

    # Encode categorical columns
    encoders = {}
    for col in ['Brand', 'Fuel_Type', 'Transmission', 'Body_Type']:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        encoders[col] = le

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(df_copy[features])
    df_copy['Cluster'] = kmeans.labels_

    # Save model and encoders
    model_dir = os.path.join(settings.BASE_DIR, 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(model_dir, 'kmeans_model.pkl'))
    joblib.dump(encoders, os.path.join(model_dir, 'encoders.pkl'))

    return df_copy, kmeans, encoders

def load_kmeans_model():
    model_path = os.path.join(settings.BASE_DIR, 'ml_models/kmeans_model.pkl')
    enc_path = os.path.join(settings.BASE_DIR, 'ml_models/encoders.pkl')

    kmeans = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    return kmeans, encoders

def recommend_cars(user_filters, scenario=None):
    df = preprocess_data(car_data)
    kmeans, encoders = load_kmeans_model()

    # Encode user input
    try:
        brand_encoded = encoders['Brand'].transform([user_filters['brand'].upper()])[0]
        fuel_encoded = encoders['Fuel_Type'].transform([user_filters['fuel_type'].upper()])[0]
        trans_encoded = encoders['Transmission'].transform([user_filters['transmission'].upper()])[0]
        body_encoded = encoders['Body_Type'].transform([user_filters['body_type'].upper()])[0]
    except:
        return []

    user_input = [[
        (user_filters['min_price'] + user_filters['max_price']) / 2,
        brand_encoded,
        fuel_encoded,
        trans_encoded,
        body_encoded,
        (user_filters['min_year'] + user_filters['max_year']) / 2
    ]]

    # Predict cluster for this input
    cluster_pred = kmeans.predict(user_input)[0]

    # Filter matching cars from the same cluster
    df['Cluster'] = kmeans.labels_
    filtered_df = df[
        (df['Cluster'] == cluster_pred) &
        (df['Price'] >= user_filters['min_price']) &
        (df['Price'] <= user_filters['max_price']) &
        (df['Year'] >= user_filters['min_year']) &
        (df['Year'] <= user_filters['max_year']) &
        (df['Brand'] == user_filters['brand'].upper())
    ]

    # Scenario adjustments
    if scenario == 'economy_downturn':
        filtered_df['Price'] *= 0.9
    elif scenario == 'ev_rise':
        filtered_df = filtered_df[filtered_df['Fuel_Type'] == 'ELECTRIC']
    elif scenario == 'gas_price_surge':
        filtered_df = filtered_df[filtered_df['Fuel_Type'] != 'PETROL']

    return filtered_df[[
        'Brand', 'Model', 'Year', 'Transmission', 'Body_Type', 'Fuel_Type', 'Price'
    ]].drop_duplicates().sort_values(by=['Year', 'Price'], ascending=[False, True]).head(7).to_dict(orient='records')

def get_form_choices():
    df = preprocess_data(car_data)
    return {
        'brands': sorted(df['Brand'].dropna().unique()),
        'fuel_types': sorted(df['Fuel_Type'].dropna().unique()),
        'transmissions': sorted(df['Transmission'].dropna().unique()),
        'body_types': sorted(df['Body_Type'].dropna().unique()),
    }
