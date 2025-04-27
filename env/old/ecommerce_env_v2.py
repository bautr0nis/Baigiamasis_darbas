import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

class EcommercePricingEnv(gym.Env):
    def __init__(self, data_path='data/synthetic_olist_data.csv', episode_length=100, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()

        self.data["product_category_name"] = self.data["product_category_name"].fillna("unknown")

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = self.encoder.fit_transform(self.data[['product_category_name']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(['product_category_name']))

        self.scaler = MinMaxScaler()
        self.data['price_scaled'] = self.scaler.fit_transform(self.data[['price']])
        self.features = pd.concat([self.data[['price_scaled', 'order_month', 'order_day']], encoded_df], axis=1)

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0
        self.action_space = spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)

        self.demand_model = joblib.load("demand_model_simulated/demand_model.pkl")
        self.category_encoder = joblib.load("demand_model_simulated/category_encoder.pkl")
        self.next_reward = 0.0

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.next_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= self.max_steps:
            return np.zeros(self.features.shape[1], dtype=np.float32)
        return self.features.iloc[self.current_step].values.astype(np.float32)

    def encode_category(self, category):
        df = pd.DataFrame({"product_category_name": [category]})
        try:
            cat_array = self.category_encoder.transform(df)
        except:
            df = pd.DataFrame({"product_category_name": ["unknown"]})
            cat_array = self.category_encoder.transform(df)
        return pd.DataFrame(cat_array, columns=self.category_encoder.get_feature_names_out(["product_category_name"]))

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0.0, True, False, {}

        row = self.data.iloc[self.current_step]
        base_price = row['price']
        cost = max(row['cost'], 0)
        freight = row['freight_value']
        review_score = row['review_score']
        delay_days = row['delivery_delay_days']
        category = row['product_category_name']
        price_change_pct = float(np.clip(action[0], -0.2, 0.2))
        new_price = base_price * (1.0 + price_change_pct)

        cat_df = self.encode_category(category)

        today_features = pd.DataFrame([{
            "price": new_price,
            "base_price": row["base_price"],
            "price_elasticity": row["price_elasticity"],
            "base_demand": row["base_demand"],
            "order_month": row["order_month"],
            "order_day": row["order_day"],
            "customer_order_count": row["customer_order_count"]
        }])
        model_input = pd.concat([today_features.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
        model_input = model_input[self.demand_model.feature_names_in_]

        try:
            log_qty = self.demand_model.predict(model_input)[0]
            predicted_qty = max(1, int(np.expm1(log_qty)))
        except:
            predicted_qty = 1

        predicted_qty = np.clip(predicted_qty, 1, 10)
        margin = new_price - cost - freight
        profit = margin * predicted_qty
        reward = profit

        if margin < 0:
            reward -= 10 * abs(margin)
        if review_score >= 4:
            reward += 2.0
        elif review_score <= 2:
            reward -= 2.0
        if delay_days > 3:
            reward -= 1.0
        if predicted_qty > row["base_demand"] * 1.2:
            reward -= 30
        if 0.9 * row["base_price"] <= new_price <= 1.1 * row["base_price"]:
            reward += 5

        # === 🔁 Add future reward signal (next day) ===
        if self.current_step + 1 < self.max_steps:
            next_row = self.data.iloc[self.current_step + 1]
            next_cat_df = self.encode_category(next_row["product_category_name"])
            next_features = pd.DataFrame([{
                "price": next_row["price"],
                "base_price": next_row["base_price"],
                "price_elasticity": next_row["price_elasticity"],
                "base_demand": next_row["base_demand"],
                "order_month": next_row["order_month"],
                "order_day": next_row["order_day"],
                "customer_order_count": next_row["customer_order_count"]
            }])
            next_input = pd.concat([next_features.reset_index(drop=True), next_cat_df.reset_index(drop=True)], axis=1)
            next_input = next_input[self.demand_model.feature_names_in_]

            try:
                next_pred_log = self.demand_model.predict(next_input)[0]
                next_qty = max(1, int(np.expm1(next_pred_log)))
            except:
                next_qty = 1

            next_margin = next_row["price"] - max(next_row["cost"], 0) - next_row["freight_value"]
            self.next_reward = next_margin * next_qty
        else:
            self.next_reward = 0.0

        final_reward = 0.7 * reward + 0.3 * self.next_reward

        self.current_step += 1
        done = self.current_step % self.episode_length == 0 or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "price": base_price,
            "cost": cost,
            "new_price": new_price,
            "freight_value": freight,
            "quantity_sold": predicted_qty,
            "review_score": review_score,
            "delivery_delay_days": delay_days,
            "reward": final_reward,
            "product_category_name": category
        }

        if self.verbose:
            print(f"[Step {self.current_step}] Price: {new_price:.2f}, Qty: {predicted_qty}, Reward: {final_reward:.2f}")

        return self._get_obs(), final_reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")