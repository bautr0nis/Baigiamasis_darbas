# train_demand_model_simulated.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
import os

# === 1. Load updated dataset ===
df = pd.read_csv("data/merged_olist_data_simulated.csv")

# === 2. Target: Simulated demand ===
df = df[df["simulated_quantity_fixed"].notna()]  # Ensure no NaNs
y = np.log1p(df["simulated_quantity_fixed"])  # log transform for stability

# === 3. Feature Engineering ===
X_raw = df[[
    "price", "order_month", "order_day",
    "product_category_name",  # categorical
    "product_photos_qty", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
    "customer_order_count"
]].copy()

# === 4. Encode category ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X_raw[["product_category_name"]])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["product_category_name"]))

# Combine with numeric features
X_final = pd.concat([
    X_raw.drop(columns=["product_category_name"]).reset_index(drop=True),
    X_encoded_df.reset_index(drop=True)
], axis=1)

# === 5. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === 6. Train model ===
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("Simulated model R2 score on test set: {:.3f}".format(r2))

# === 8. Save model and encoder ===
os.makedirs("demand_model_simulated", exist_ok=True)
joblib.dump(model, "demand_model_simulated/demand_model.pkl")
joblib.dump(encoder, "demand_model_simulated/category_encoder.pkl")
print("Model & encoder saved to 'demand_model_simulated/'")
