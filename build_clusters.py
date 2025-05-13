import os
import django
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'autoinsight360.settings')
django.setup()

from dashboard.recommend import preprocess_data

# Load and preprocess data
df = pd.read_csv('dashboard/car_data.csv')
df = preprocess_data(df)

# Encode categorical features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()
le_body = LabelEncoder()

df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel_Type'] = le_fuel.fit_transform(df['Fuel_Type'])
df['Transmission'] = le_trans.fit_transform(df['Transmission'])
df['Body_Type'] = le_body.fit_transform(df['Body_Type'])

X = df[['Price', 'Brand', 'Fuel_Type', 'Transmission', 'Body_Type', 'Year']]

# Grid Search over range of K
inertia_values = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

# Select best K based on elbow
best_k = inertia_values.index(min(inertia_values)) + 2  # because range starts at 2
print(f"✅ Best K found: {best_k}")

# Retrain final model
final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_model.fit(X)
df['Cluster'] = final_model.labels_

# Save model and encoders
joblib.dump(final_model, 'ml_models/kmeans_model.pkl')
joblib.dump({
    'Brand': le_brand,
    'Fuel_Type': le_fuel,
    'Transmission': le_trans,
    'Body_Type': le_body
}, 'ml_models/encoders.pkl')

print("✅ Model and encoders saved.")
