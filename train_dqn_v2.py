# train_dqn.py
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.old.ecommerce_env_analytical import EcommercePricingEnv
import os

# === 1. Setup ===
RUN_NAME = "DQN_run_discrete"
model_dir = f"models/dqn/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)

# === 2. Environment ===
env = DummyVecEnv([
    lambda: Monitor(EcommercePricingEnv(
        data_path="data/synthetic_olist_data.csv",
        verbose=False
    ))
])

# === 3. DQN Model ===
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    learning_rate=1e-4,
    buffer_size=50000,
    batch_size=64,
    learning_starts=1000,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05
)

# === 4. Train ===
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME)

# === 5. Save ===
model.save(f"{model_dir}/dqn_pricing_model")
print(f"✅ DQN model saved to {model_dir}/dqn_pricing_model")