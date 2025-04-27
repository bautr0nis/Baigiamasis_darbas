from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.old.ecommerce_env_simple2 import AdvancedPricingEnv
import os

# === 1. Setup ===
RUN_NAME = "DQN_run_advanced"
model_dir = f"models/dqn/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

# === 2. Environment ===
env = DummyVecEnv([
    lambda: Monitor(
        AdvancedPricingEnv(data_path="data/synthetic_olist_data2.csv", verbose=False),
        filename="logs/dqn/monitor.csv"
    )
])

# === 3. DQN model ===
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# === 4. Train the model ===
TIMESTEPS = 300_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME)

# === 5. Save the model ===
model.save(f"{model_dir}/dqn_pricing_model")
print(f"✅ DQN advanced model saved to {model_dir}/dqn_pricing_model")