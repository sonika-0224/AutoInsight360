import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load original data
df = pd.read_csv('car_data.csv')

# Keep necessary features
df = df[['MSRP', 'Year', 'Engine HP', 'Engine Cylinders', 'Engine Fuel Type', 'Transmission Type', 'Vehicle Size']].dropna()

# Encode categorical features
df['Engine Fuel Type'] = df['Engine Fuel Type'].astype('category').cat.codes
df['Transmission Type'] = df['Transmission Type'].astype('category').cat.codes
df['Vehicle Size'] = df['Vehicle Size'].astype('category').cat.codes

# Simulate scenario impact on MSRP
def simulate_scenario(row, scenario):
    base_price = row['MSRP']
    if scenario == 'economy_downturn':
        return base_price * 0.9
    elif scenario == 'ev_rise':
        return base_price * 1.05 if row['Engine Fuel Type'] == 2 else base_price
    elif scenario == 'gas_price_surge':
        return base_price * 1.07 if row['Engine Fuel Type'] == 0 else base_price
    else:
        return base_price

# Create multiple scenario rows
df_all = pd.concat([
    df.assign(Scenario='none', Adjusted_Price=df['MSRP']),
    df.assign(Scenario='economy_downturn', Adjusted_Price=df.apply(lambda row: simulate_scenario(row, 'economy_downturn'), axis=1)),
    df.assign(Scenario='ev_rise', Adjusted_Price=df.apply(lambda row: simulate_scenario(row, 'ev_rise'), axis=1)),
    df.assign(Scenario='gas_price_surge', Adjusted_Price=df.apply(lambda row: simulate_scenario(row, 'gas_price_surge'), axis=1)),
])

# Encode scenario
df_all['Scenario'] = df_all['Scenario'].astype('category').cat.codes

# Features and target
X = df_all[['Year', 'Engine HP', 'Engine Cylinders', 'Engine Fuel Type', 'Transmission Type', 'Vehicle Size', 'Scenario']]
y = df_all['Adjusted_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ml_models/scenario_regressor.pkl')
print("✅ Scenario regressor saved!")
