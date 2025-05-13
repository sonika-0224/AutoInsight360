import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load and clean data
def load_clean_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['MSRP', 'Year', 'Make', 'Model', 'Transmission Type', 'Vehicle Size', 'Engine HP'])
    df = df[df['MSRP'] < 100000]  # Optional: remove luxury outliers
    return df

# Train KMeans model on key numeric features
def train_model(df):
    features = df[['MSRP', 'Engine HP', 'Year']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = model.fit_predict(X_scaled)
    return model, scaler, df

# Predict best cluster based on inputs
def predict_cluster(model, scaler, input_data):
    scaled = scaler.transform([input_data])
    return model.predict(scaled)[0]
