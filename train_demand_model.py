import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Įkeliame duomenis
df = pd.read_csv("data/merged_olist_data.csv")

# Imituojam quantity_sold = 1 kiekvienam užsakymui
df["quantity"] = 1

# Grupavimas pagal produktą + laiką
agg = df.groupby(["product_id", "product_category_name", "order_month", "order_day"]).agg({
    "price": "mean",
    "quantity": "sum"
}).reset_index()

# One-hot kategorija
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(agg[["product_category_name"]])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# X ir y
X = pd.concat([agg[["price", "order_month", "order_day"]], encoded_df], axis=1)
y = agg["quantity"]

# Modelio treniravimas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Išsaugom modelį ir encoderį
os.makedirs("demand_model", exist_ok=True)
joblib.dump(model, "demand_model/demand_model.pkl")
joblib.dump(encoder, "demand_model/category_encoder.pkl")
print("✅ Demand model & encoder saved.")