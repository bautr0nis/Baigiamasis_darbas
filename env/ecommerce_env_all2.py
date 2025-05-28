import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import OneHotEncoder

class AdvancedPricingEnv(gym.Env):
    def __init__(self, data_path='data/generated/weekly_env_data_filled.csv', episode_length=52, verbose=False):
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

        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(7 + self.encoded_categories.shape[1],),
            dtype=np.float32
        )

        self.trajectory_log = []

    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.prev_quantity_sold = None
        self.previous_reward = 0.0
        self.previous_price = None
        self.consecutive_price_increases = 0
        self.trajectory_log = []
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
            demand = base_demand * (ratio ** (elasticity * 3))
            demand *= np.random.uniform(0.95, 1.05)
            demand = max(1, round(demand))
        except Exception as e:
            print(f"KLAIDA skaičiuojant demand ({e}), step {self.current_step}")
            demand = 1

        effective_sales = min(demand, stock)
        margin = new_price - cost
        profit = margin * effective_sales
        reward = profit

        conversion_rate = effective_sales / demand if demand > 0 else 0
        stock_left = stock - effective_sales

        reward_components = {
            "profit": profit,
            "bonus_full_stock_sold": 50 if stock_left == 0 else 0,
            "penalty_stock_accumulation": -1.5 * stock_left if stock_left > 0.3 * stock else 0,
            "bonus_high_conversion": 10 if conversion_rate > 0.7 else 0,
            "penalty_low_sales_vs_stock": -20 if effective_sales < 0.5 * stock else 0,
            "bonus_volume": 0.05 * effective_sales,
            "penalty_demand_drop": -10 if demand < 0.8 * base_demand else 0,
            "penalty_high_price_low_demand": -10 if demand < 0.5 * base_demand and new_price > base_price else 0,
            "bonus_discount_under_low_demand": 10 if demand < 5 and price_change_pct < 0 else 0,
            "adjustment_low_prev_demand_and_price_change":
                (-10 if price_change_pct > 0 else 5)
                if self.prev_quantity_sold is not None and self.prev_quantity_sold < 0.8 * base_demand else 0,
            "penalty_price_fatigue": -15 if self.consecutive_price_increases >= 3 else 0,
            "bonus_aggressive_stock_clear": 0.25 * profit if stock > 30 and price_change_pct < 0 and demand > 0.6 * base_demand else 0
        }

        reward += sum(reward_components.values()) - profit

        if self.previous_price is not None and new_price > self.previous_price:
            self.consecutive_price_increases += 1
        else:
            self.consecutive_price_increases = 0

        self.previous_price = new_price
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
            "year_week": row.get("year_week", ""),
            "action_taken": action,
            "price_change_pct": price_change_pct,
            "reward_components": reward_components
        }

        self.trajectory_log.append({
            "step": self.current_step,
            "action": action,
            "obs": self._get_obs().tolist(),
            "reward": reward,
            "info": info
        })

        if self.verbose:
            print(f"[Step {self.current_step}] Price: {new_price:.2f}, Sold: {effective_sales}, Reward: {reward:.2f}")
            for k, v in reward_components.items():
                print(f"   - {k}: {v:.2f}")

        return self._get_obs(), reward, done, truncated, info

    def render(self):
        print(f"Step: {self.current_step}")

    def get_trajectory_log(self):
        return self.trajectory_log
