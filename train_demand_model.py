import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import joblib
import os

# === 1. Load dataset ===
df = pd.read_csv("data/merged_olist_data.csv")

# === 2. Target quantity ===
df["quantity"] = df.groupby(["order_id", "product_id"])["order_item_id"].transform("count")

# === 3. Date prep ===
df["order_date"] = pd.to_datetime(df["order_purchase_timestamp"])
df = df.sort_values("order_date")

# === 4. Rolling demand & monthly avg ===
df["rolling_quantity"] = df.groupby("product_id")["quantity"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
df["month_avg_sales"] = df.groupby(["product_category_name", "order_month"])["quantity"].transform("mean")

# === 5. Fill NA ===
df["product_category_name"] = df["product_category_name"].fillna("unknown")
df["rolling_quantity"] = df["rolling_quantity"].fillna(1)
df["month_avg_sales"] = df["month_avg_sales"].fillna(1)
df["customer_order_count"] = df["customer_order_count"].fillna(1)

# === 6. Features ===
X_raw = df[[
    "price", "order_month", "order_day",
    "product_category_name",  # tik ši kategorija
    "product_photos_qty", "product_weight_g",
    "product_length_cm", "product_height_cm", "product_width_cm",
    "month_avg_sales", "rolling_quantity",
    "customer_order_count"
]].copy()

y = np.log1p(df["quantity"])

# === 7. Encode category ===
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_encoded = encoder.fit_transform(X_raw[["product_category_name"]])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(["product_category_name"]))
X_final = pd.concat([
    X_raw.drop(columns=["product_category_name"]).reset_index(drop=True),
    X_encoded_df.reset_index(drop=True)
], axis=1)

# === 8. Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# === 9. Train model ===
model = HistGradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# === 10. Evaluate ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"✅ Model R² score on test set: {r2:.3f}")

# === 11. Save model and encoder ===
os.makedirs("demand_model", exist_ok=True)
joblib.dump(model, "demand_model/demand_model.pkl")
joblib.dump(encoder, "demand_model/category_encoder.pkl")
print("📦 Demand model and encoder saved to 'demand_model/'")