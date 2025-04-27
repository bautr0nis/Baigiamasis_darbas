import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder

class AdvancedPricingEnv(gym.Env):
    def __init__(self, data_path='data/weekly_env_data.csv', episode_length=50, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()

        self.data["translated_name"] = self.data["translated_name"].fillna("unknown")
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoded_categories = self.encoder.fit_transform(self.data[["translated_name"]])

        self.max_steps = len(self.data)
        self.episode_length = episode_length
        self.current_step = 0

        self.action_space = spaces.Discrete(9)  # -20% to +20%, step 5%

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(7 + self.encoded_categories.shape[1],),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_quantity_sold = None
        self.previous_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        category_vec = self.encoded_categories[self.current_step]

        demand_level = 1
        if self.prev_quantity_sold is not None:
            base_demand = row['base_demand']
            if self.prev_quantity_sold < 0.9 * base_demand:
                demand_level = 0
            elif self.prev_quantity_sold > 1.1 * base_demand:
                demand_level = 2

        obs = np.concatenate([
            np.array([
                row['avg_price'],
                row['base_price'],
                row['base_demand'],
                row['price_elasticity'],
                row['stock'],
                self.previous_reward,
                demand_level
            ], dtype=np.float32),
            category_vec.astype(np.float32)
        ])
        return obs

    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_obs(), 0.0, True, False, {}

        row = self.data.iloc[self.current_step]
        base_price = row['avg_price']
        base_demand = row['base_demand']
        elasticity = row['price_elasticity']
        cost = row['avg_cost']
        stock = row['stock']

        price_change_pct = (action - 4) * 0.05
        new_price = base_price * (1 + price_change_pct)

        try:
            ratio = new_price / base_price if base_price != 0 else 1
            elasticity = elasticity if not pd.isna(elasticity) else -1.5
            demand = base_demand * (ratio ** (elasticity * 2))
            demand *= np.random.uniform(0.9, 1.1)
            demand = max(1, round(demand))
        except Exception as e:
            print(f"⚠️ KLAIDA skaičiuojant demand ({e}), step {self.current_step}")
            demand = 1

        effective_sales = min(demand, stock)
        margin = new_price - cost
        profit = margin * effective_sales
        reward = profit

        # === Business logic penalties/bonuses ===
        if demand > stock:
            reward -= (demand - stock) * 2  # Lost sales penalty

        if demand < 0.5 * base_demand:
            reward -= (base_demand - demand) * 2.0

        if self.prev_quantity_sold:
            drop = self.prev_quantity_sold - demand
            if drop > 0.2 * self.prev_quantity_sold:
                reward -= 0.5 * drop

        if margin < 0:
            reward -= 10 * abs(margin)

        if price_change_pct > 0.1:
            reward -= 5

        if demand < 0.8 * base_demand and price_change_pct > 0:
            reward -= 5

        if self.prev_quantity_sold is not None and self.prev_quantity_sold < 0.8 * base_demand:
            if price_change_pct < 0:
                reward += 5  # Incentivize price drops when demand is low

        self.previous_reward = reward
        self.prev_quantity_sold = demand
        self.current_step += 1

        done = self.current_step % self.episode_length == 0 or self.current_step >= self.max_steps
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "price": base_price,
            "cost": cost,
            "new_price": new_price,
            "quantity_sold": effective_sales,
            "total_demand": demand,
            "stock": stock,
            "reward": reward,
            "price_elasticity": row["price_elasticity"],
            "product_category_name": row.get("translated_name", "unknown"),
            "year_week": row.get("year_week", "")
        }

        if self.verbose:
            print(f"[Step {self.current_step}] Price: {new_price:.2f}, Sold: {effective_sales}, Reward: {reward:.2f}")

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")