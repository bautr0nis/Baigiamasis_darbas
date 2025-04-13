import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and encoder
model = joblib.load("demand_model/demand_model.pkl")
encoder = joblib.load("demand_model/category_encoder.pkl")

# Create base input
base_input = {
    "order_month": 6,
    "order_day": 15,
    "product_category_name": "furniture_decor",
    "product_photos_qty": 1,
    "product_weight_g": 2000,
    "product_length_cm": 30,
    "product_height_cm": 10,
    "product_width_cm": 15,
    "month_avg_sales": 3,
    "rolling_quantity": 2,
    "customer_order_count": 4
}

prices = np.linspace(80, 120, 20)
predicted_quantities = []

# Encode category once
cat_encoded = encoder.transform([[base_input["product_category_name"]]])
cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["product_category_name"]))

for price in prices:
    row = pd.DataFrame([{
        "price": price,
        "order_month": base_input["order_month"],
        "order_day": base_input["order_day"],
        "product_photos_qty": base_input["product_photos_qty"],
        "product_weight_g": base_input["product_weight_g"],
        "product_length_cm": base_input["product_length_cm"],
        "product_height_cm": base_input["product_height_cm"],
        "product_width_cm": base_input["product_width_cm"],
        "month_avg_sales": base_input["month_avg_sales"],
        "rolling_quantity": base_input["rolling_quantity"],
        "customer_order_count": base_input["customer_order_count"]
    }])

    X_full = pd.concat([row.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
    X_full = X_full[model.feature_names_in_]  # ⬅️ labai svarbu: seka tokia pati kaip modelio!

    pred_log = model.predict(X_full)[0]
    predicted_quantities.append(np.expm1(pred_log))

# 📈 Plot
plt.figure(figsize=(8, 5))
plt.plot(prices, predicted_quantities, marker='o')
plt.title("📉 Predicted Quantity vs Price")
plt.xlabel("Price")
plt.ylabel("Predicted Quantity")
plt.grid(True)
plt.tight_layout()
plt.show()