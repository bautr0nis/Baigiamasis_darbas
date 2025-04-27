import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder

class EcommercePricingEnv(gym.Env):
    def __init__(self, data_path='data/synthetic_olist_data.csv', episode_length=100, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()

        # === One-hot encode category ===
        self.data["product_category_name"] = self.data["product_category_name"].fillna("unknown")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoded_categories = self.encoder.fit_transform(self.data[["product_category_name"]])

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0

        self.action_space = spaces.Discrete(41)  # -20% to +20% in 1% steps
        # 6 numeric features + len(category vector)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(6 + self.encoded_categories.shape[1],),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_quantity_sold = None
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        category_vec = self.encoded_categories[self.current_step]
        obs = np.concatenate([
            np.array([
                row['price'],
                row['base_price'],
                row['price_elasticity'],
                row['base_demand'],
                row['order_month'],
                row['order_day']
            ], dtype=np.float32),
            category_vec.astype(np.float32)
        ])
        return obs

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0.0, True, False, {}

        row = self.data.iloc[self.current_step]
        base_price = row['price']
        base_demand = row['base_demand']
        elasticity = row['price_elasticity']
        cost = max(row['cost'], 0)

        # Convert discrete action to percent change
        price_change_pct = (action - 20) / 100.0  # from -0.20 to +0.20
        new_price = base_price * (1.0 + price_change_pct)

        # === Analytical demand ===
        demand = base_demand * (new_price / base_price) ** (elasticity * 2)
        demand = np.clip(round(demand), 1, 10)
        margin = new_price - cost
        profit = margin * demand
        reward = profit

        # === Penalties / bonuses ===
        if demand < 0.5 * base_demand:
            drop_penalty = (base_demand - demand) * 2.0
            reward -= drop_penalty

        if self.prev_quantity_sold:
            drop = self.prev_quantity_sold - demand
            if drop > 0.2 * self.prev_quantity_sold:
                reward -= 0.5 * drop
        self.prev_quantity_sold = demand

        if margin < 0:
            reward -= 10 * abs(margin)

        if 0.9 * row['base_price'] <= new_price <= 1.1 * row['base_price'] and demand >= base_demand:
            reward += 5.0

        self.current_step += 1
        done = self.current_step % self.episode_length == 0 or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "price": base_price,
            "cost": cost,
            "new_price": new_price,
            "quantity_sold": demand,
            "reward": reward,
            "product_category_name": row.get("product_category_name", "unknown")
        }

        if self.verbose:
            print(f"[Step {self.current_step}] 💵 Price: {new_price:.2f}, Qty: {demand}, Reward: {reward:.2f}")

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")