# what_if_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from stable_baselines3 import DQN
from env.ecommerce_env_all2 import AdvancedPricingEnv
from gymnasium import spaces

# === 1. Setup ===
model_path = "models/dqn/DQN_run_real/dqn_pricing_model"
audit_path = "data/generated/eval_output_dqn_real.csv"
data_path = "data/generated/weekly_env_data_filled.csv"

# === 2. Load model and full data ===
model = DQN.load(model_path)
full_data = pd.read_csv(data_path)
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")
df_eval = pd.read_csv(audit_path)



# === 3. Simulate all actions ===
def simulate_all_actions(env, obs):
    results = []
    for a in range(env.action_space.n):
        env_copy = copy.deepcopy(env)
        env_copy.current_step -= 1
        sim_obs, reward, _, _, info = env_copy.step(a)

        reward_components = info.get("reward_components", {})
        profit = reward_components.get("profit", np.nan)

        results.append({
            "action": a,
            "sim_reward": reward,
            "sim_profit": profit,
            "sim_sales": info.get("quantity_sold", np.nan),
            "sim_demand": info.get("total_demand", np.nan),
            "price_elasticity": info.get("price_elasticity", np.nan),
            "stock": info.get("stock", np.nan),
            "category": info.get("product_category_name", "unknown")
        })
    return pd.DataFrame(results)

# === 4. Run what-if analysis ===
policy_vs_best = []

for i, row in df_eval.iterrows():
    current_category = row["product_category_name"]
    df_cat = full_data[full_data["translated_name"] == current_category].reset_index(drop=True)

    env = AdvancedPricingEnv(data_path=data_path, verbose=False)
    env.data = df_cat
    env.original_data = df_cat
    env.max_steps = len(df_cat)
    env.current_step = min(row["step"], len(df_cat) - 1)

    env.encoder = env.encoder.fit(df_cat[["translated_name"]])
    env.encoded_categories = env.encoder.transform(df_cat[["translated_name"]])
    env.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(7 + env.encoded_categories.shape[1],),
        dtype=np.float32
    )

    obs = env._get_obs()
    action_taken = row["action"]

    sim_df = simulate_all_actions(env, obs)
    sim_df["step"] = row["step"]
    sim_df["actual_action"] = action_taken
    sim_df["actual_reward"] = row["reward"]

    best_row = sim_df.loc[sim_df["sim_reward"].idxmax()]
    regret = best_row["sim_reward"] - sim_df.loc[sim_df["action"] == action_taken, "sim_reward"].values[0]
    sim_df["regret"] = regret
    sim_df["best_action"] = best_row["action"]

    policy_vs_best.append(sim_df)

# === 5. Save output ===
full_df = pd.concat(policy_vs_best)
os.makedirs("data/generated", exist_ok=True)
full_df.to_csv("data/generated/what_if_analysis.csv", index=False)
print("✅ Saved what-if action simulation to data/generated/what_if_analysis.csv")




