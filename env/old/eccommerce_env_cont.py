import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib

class EcommercePricingEnv(gym.Env):
    def __init__(self, data_path='data/merged_olist_data.csv', episode_length=100, verbose=False):
        super(EcommercePricingEnv, self).__init__()

        self.verbose = verbose
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()

        self.data["product_category_name"] = self.data["product_category_name"].fillna("unknown")

        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = self.encoder.fit_transform(self.data[['product_category_name']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())

        self.scaler = MinMaxScaler()
        self.data['price_scaled'] = self.scaler.fit_transform(self.data[['price']])

        self.features = pd.concat([
            self.data[['price_scaled', 'order_month', 'order_day']],
            encoded_df
        ], axis=1)

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0
        self.prev_price = None
        self.prev_predicted_quantity = None

        self.action_space = spaces.Box(low=np.array([-0.2]), high=np.array([0.2]), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        self.demand_model = joblib.load("demand_model_simulated/demand_model.pkl")
        self.category_encoder = joblib.load("demand_model_simulated/category_encoder.pkl")

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_price = None
        self.prev_predicted_quantity = None
        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step >= self.max_steps:
            return np.zeros(self.features.shape[1], dtype=np.float32)
        return self.features.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0.0, True, False, {}

        row = self.data.iloc[self.current_step]
        base_price = row['price']
        freight = row['freight_value']
        cost = max(row['cost'], 0)  # prevent negative cost
        review_score = row['review_score']
        delay_days = row['delivery_delay_days']
        product_id = row['product_id']

        price_change_pct = float(np.clip(action[0], -0.2, 0.2))
        new_price = base_price * (1.0 + price_change_pct)

        category_name = row['product_category_name']
        if pd.isnull(category_name) or not isinstance(category_name, str):
            category_name = "unknown"

        try:
            category_input_df = pd.DataFrame({"product_category_name": [category_name]})
            category_array = self.category_encoder.transform(category_input_df)
        except Exception:
            category_array = self.category_encoder.transform([["unknown"]])

        category_df = pd.DataFrame(category_array, columns=self.category_encoder.get_feature_names_out())
        features = pd.DataFrame({
            "price": [new_price],
            "order_month": [row['order_month']],
            "order_day": [row['order_day']]
        })
        features = pd.concat([features, category_df], axis=1)

        # Simulated demand response (clipped normal)
        predicted_quantity = np.clip(np.random.normal(2.5, 1.0), 1, 5)
        quantity_sold = int(round(predicted_quantity))

        # Profit calculation
        margin = new_price - cost - freight
        profit = margin * quantity_sold

        # Penalty if loss
        if margin < 0:
            profit -= 50.0  # soften penalty for under-margin

        # Bonus/penalty for external factors
        if review_score >= 4.0:
            profit += 2.0
        elif review_score <= 2.0:
            profit -= 2.0
        if delay_days > 3:
            profit -= 1.0

        reward = profit

        # Penalize sharp demand drop
        if self.prev_price is not None and self.prev_predicted_quantity is not None:
            quantity_drop = self.prev_predicted_quantity - predicted_quantity
            if quantity_drop > 4:
                reward -= 0.5 * quantity_drop

        self.prev_price = new_price
        self.prev_predicted_quantity = predicted_quantity

        self.current_step += 1
        done = self.current_step % self.episode_length == 0 or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps

        info = {
            "product_id": product_id,
            "price": base_price,
            "cost": cost,
            "freight_value": freight,
            "quantity_sold": quantity_sold,
            "new_price": new_price,
            "review_score": review_score,
            "delivery_delay_days": delay_days,
            "reward": reward,
            "step": self.current_step,
            "product_category_name": category_name
        }

        if self.verbose:
            print(f"[Step {self.current_step}] Price: {new_price:.2f}, Cost: {cost:.2f}, Qty: {quantity_sold}, Profit: {profit:.2f}, Reward: {reward:.2f}")

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")