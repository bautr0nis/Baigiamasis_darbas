import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C
from env.ecommerce_env_real2 import AdvancedPricingEnv

# === 1. Setup ===
MODELS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C
}

RUN_NAME = "real2"
MODEL_DIR = f"models/compare/{RUN_NAME}"
OUTPUT_DIR = f"data/generated/eval_outputs/{RUN_NAME}"
ANALYSIS_DIR = f"analysis/compare/{RUN_NAME}"
LOG_DIR = f"logs/compare/{RUN_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# === 2. Load full data once ===
full_data = pd.read_csv("data/generated/weekly_env_data_filled.csv")

# === 3. Evaluate each model ===
for model_name, model_class in MODELS.items():
    print(f"\n🚀 Evaluating {model_name}...")

    model_path = f"{MODEL_DIR}/{model_name}_pricing_model"
    model = model_class.load(model_path)

    # Create a full env to reuse encoder
    full_env = AdvancedPricingEnv(data_path="data/generated/weekly_env_data_filled.csv", verbose=False)

    all_steps = []  # Collect steps from all categories

    for category, df_cat in full_data.groupby('translated_name'):
        print(f"🔍 Evaluating category: {category}...")

        # Create temporary env manually
        env = AdvancedPricingEnv(data_path="data/generated/weekly_env_data_filled.csv", verbose=False)

        # Overwrite env data
        env.data = df_cat.reset_index(drop=True)
        env.original_data = df_cat.reset_index(drop=True)
        env.max_steps = len(df_cat)
        env.current_step = 0

        # Reuse the encoder
        env.encoder = full_env.encoder
        env.encoded_categories = env.encoder.transform(env.data[["translated_name"]])

        # Create new correct observation space
        env.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7 + env.encoded_categories.shape[1],),
            dtype=np.float32
        )

        obs = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            info["action"] = int(action)
            info["category"] = category
            total_reward += reward
            all_steps.append(info)

    # Save combined output
    df_result = pd.DataFrame(all_steps)
    df_result.to_csv(f"{OUTPUT_DIR}/eval_output_{model_name}.csv", index=False)

    print(f"✅ {model_name} Total Reward (all categories combined): {total_reward:.2f}")

# === 4. Plot training curves ===
rolling_window = 50
reward_data = {}

for model_name in MODELS.keys():
    monitor_path = f"{LOG_DIR}/{model_name}.monitor.csv"
    try:
        df_monitor = pd.read_csv(monitor_path, skiprows=1)
        df_monitor['rolling_reward'] = df_monitor['r'].rolling(window=rolling_window).mean()
        reward_data[model_name] = df_monitor['rolling_reward'].reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Could not load monitor for {model_name}: {e}")

plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

for model_name, rewards in reward_data.items():
    sns.lineplot(data=rewards, label=model_name)

plt.title(f"RL Models Rolling Average Reward Comparison ({RUN_NAME})")
plt.xlabel("Episode")
plt.ylabel(f"Rolling Average Reward (window={rolling_window})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/all_models_training_comparison.png")
plt.close()

print(f"\n✅ Evaluation outputs and training curves saved in {OUTPUT_DIR} and {ANALYSIS_DIR}")