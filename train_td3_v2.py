from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.old.ecommerce_env_v2 import EcommercePricingEnv
import numpy as np
import os

# === 1. Setup ===
RUN_NAME = "TD3_run_synthetic"
model_dir = f"models/td3/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)

# === 2. Environment ===
env = DummyVecEnv([lambda: Monitor(EcommercePricingEnv(data_path="data/synthetic_olist_data.csv"))])

# === 3. Action noise for better exploration ===
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# === 4. TD3 model setup ===
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./tensorboard/",
    learning_rate=1e-4,
    buffer_size=100000,
    batch_size=64,
    learning_starts=1000,
    train_freq=(1, "step"),
    gradient_steps=1,
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5
)

# === 5. Train the agent ===
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME)

# === 6. Save the model ===
model.save(f"{model_dir}/td3_pricing_model")
print(f"✅ TD3 model saved to {model_dir}/td3_pricing_model")