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

        # Fill missing category names
        self.data["product_category_name"] = self.data["product_category_name"].fillna("unknown")

        # Encode product categories
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = self.encoder.fit_transform(self.data[['product_category_name']])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out())

        # Normalize price
        self.scaler = MinMaxScaler()
        self.data['price_scaled'] = self.scaler.fit_transform(self.data[['price']])

        # Combine features
        self.features = pd.concat([
            self.data[['price_scaled', 'order_month', 'order_day']],
            encoded_df
        ], axis=1)

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0
        self.prev_action = None  # Track previous action

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.features.shape[1],),
            dtype=np.float32
        )

        # Demand prediction model
        self.demand_model = joblib.load("demand_model/demand_model.pkl")
        self.category_encoder = joblib.load("demand_model/category_encoder.pkl")

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_action = None  # Reset previous action
        obs = self._get_obs()
        return obs, {}

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
        cost = row['cost']
        review_score = row['review_score']
        delay_days = row['delivery_delay_days']
        product_id = row['product_id']

        # Apply action to price
        if action == 0:
            new_price = base_price * 0.9
        elif action == 1:
            new_price = base_price
        elif action == 2:
            new_price = base_price * 1.1
        else:
            raise ValueError("Invalid action")

        # Category processing
        category_name = row['product_category_name']
        if pd.isnull(category_name) or not isinstance(category_name, str):
            category_name = "unknown"

        try:
            category_input_df = pd.DataFrame({"product_category_name": [category_name]})
            category_array = self.category_encoder.transform(category_input_df)
        except Exception as e:
            if self.verbose:
                print(f"[WARNING] Encoder error with category '{category_name}': {e}")
            category_array = self.category_encoder.transform([["unknown"]])

        category_df = pd.DataFrame(category_array, columns=self.category_encoder.get_feature_names_out())
        features = pd.DataFrame({
            "price": [new_price],
            "order_month": [row['order_month']],
            "order_day": [row['order_day']]
        })
        features = pd.concat([features, category_df], axis=1)

        # Demand prediction
        try:
            predicted_quantity = self.demand_model.predict(features)[0]
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Demand prediction failed: {e}")
            predicted_quantity = 1.0

        quantity_sold = max(1, int(predicted_quantity))

        # Reward: profit + modifiers
        profit = (new_price - cost - freight) * quantity_sold

        if review_score >= 4.0:
            profit += 1.0
        elif review_score <= 2.0:
            profit -= 1.0
        if delay_days > 3:
            profit -= 0.8

        reward = profit + 100  # normalization offset

        # 💡 Policy regularization: penalize frequent price strategy switching
        if self.prev_action is not None and self.prev_action != action:
            reward -= 0.5
        self.prev_action = action

        if self.verbose:
            print(f"[Step {self.current_step}] Price={new_price:.2f}, Qty={quantity_sold}, "
                  f"Reward={reward:.2f}, Cat={category_name}, Act={action}")

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

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")