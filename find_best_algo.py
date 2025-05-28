import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.ecommerce_env_all import AdvancedPricingEnv

# === Setup ===
MODELS = {
    "DQN": DQN,
    "PPO": PPO,
    "A2C": A2C
}

RUN_NAME = "real2"
MODEL_DIR = f"models/compare/{RUN_NAME}"
LOG_DIR = f"logs/compare/{RUN_NAME}"
ANALYSIS_DIR = f"analysis/compare/{RUN_NAME}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

TIMESTEPS = 350_000

def make_env(model_name):
    monitor_path = f"{LOG_DIR}/{model_name}.monitor.csv"
    return DummyVecEnv([
        lambda: Monitor(
            AdvancedPricingEnv(data_path="data/generated/weekly_env_data_filled.csv", verbose=False),
            filename=monitor_path
        )
    ])

# === Train models ===
for model_name, model_class in MODELS.items():
    print(f"\n🚀 Training {model_name}...")
    env = make_env(model_name)  # <---- PERDUODAM modelio pavadinimą
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        gamma=0.8,
        verbose=1,
        tensorboard_log=f"tensorboard/{RUN_NAME}/{model_name}"
    )
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=model_name)
    model.save(f"{MODEL_DIR}/{model_name}_pricing_model")
    print(f"✅ {model_name} saved.")

# === Evaluate models ===
rolling_window = 50
reward_data = {}

for model_name in MODELS.keys():
    monitor_path = f"logs/compare/{RUN_NAME}/{model_name}.monitor.csv"
    # Try to load monitor file
    try:
        df = pd.read_csv(monitor_path, skiprows=1)  # Skip first comment line
        df['rolling_reward'] = df['r'].rolling(window=rolling_window).mean()
        reward_data[model_name] = df['rolling_reward'].reset_index(drop=True)
    except Exception as e:
        print(f"⚠️ Could not load monitor for {model_name}: {e}")

# === Plot ===
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

for model_name, rewards in reward_data.items():
    sns.lineplot(data=rewards, label=model_name)

#plt.title(f"RL Models Rolling Average Reward Comparison ({RUN_NAME})")
plt.xlabel("Simuliacinis laikotarpis (angl. episode)")
plt.ylabel(f"Altygis")
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/all_models_training_comparison.png")
plt.close()

print(f"\n✅ All models compared and graph saved to {ANALYSIS_DIR}/all_models_training_comparison.png")