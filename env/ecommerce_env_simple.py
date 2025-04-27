import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder

class SimplePricingEnv(gym.Env):
    def __init__(self, data_path='data/synthetic_olist_data.csv', episode_length=100, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()

        self.data["product_category_name"] = self.data["product_category_name"].fillna("unknown")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoded_categories = self.encoder.fit_transform(self.data[["product_category_name"]])

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # 0 = decrease, 1 = same, 2 = increase

        # Observation: current_price, base_price, base_demand, price_elasticity, demand_level, category (one-hot)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(5 + self.encoded_categories.shape[1],),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_quantity_sold = None
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        category_vec = self.encoded_categories[self.current_step]

        demand_level = 1  # neutral by default
        if self.prev_quantity_sold is not None:
            base_demand = row['base_demand']
            if self.prev_quantity_sold < 0.9 * base_demand:
                demand_level = 0  # low
            elif self.prev_quantity_sold > 1.1 * base_demand:
                demand_level = 2  # high

        obs = np.concatenate([
            np.array([
                row['price'],
                row['base_price'],
                row['base_demand'],
                row['price_elasticity'],
                demand_level
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

        # Action: 0 = decrease 5%, 1 = same, 2 = increase 5%
        price_change_pct = [-0.05, 0.0, 0.05][action]
        new_price = base_price * (1 + price_change_pct)

        demand = base_demand * (new_price / base_price) ** (elasticity * 1)
        demand = np.clip(round(demand), 1, 10)

        margin = new_price - cost
        profit = margin * demand
        reward = profit

        if demand < 0.5 * base_demand:
            reward -= (base_demand - demand) * 2.0

        if self.prev_quantity_sold:
            drop = self.prev_quantity_sold - demand
            if drop > 0.2 * self.prev_quantity_sold:
                reward -= 0.5 * drop
        self.prev_quantity_sold = demand

        if margin < 0:
            reward -= 10 * abs(margin)

        self.current_step += 1
        done = self.current_step % self.episode_length == 0 or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "price": base_price,
            "cost": cost,
            "new_price": new_price,
            "quantity_sold": demand,
            "price_elasticity": row["price_elasticity"],
            "reward": reward,
            "product_category_name": row.get("product_category_name", "unknown")
        }

        if self.verbose:
            print(f"[Step {self.current_step}] Price: {new_price:.2f}, Qty: {demand}, Reward: {reward:.2f}")

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")
