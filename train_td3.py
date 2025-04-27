from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env.old.ecommerce_env_analytical import EcommercePricingEnv
import numpy as np
import os

# === 1. Setup paths ===
RUN_NAME = "TD3_run_analytical"
model_dir = f"models/td3/{RUN_NAME}"
log_dir = f"logs/td3/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# === 2. Create and wrap environment ===
def make_env():
    env = EcommercePricingEnv(
        data_path="data/synthetic_olist_data.csv",
        verbose=False
    )
    env = Monitor(env)  # Logs episode reward
    return env

env = DummyVecEnv([make_env])
env = VecMonitor(env, filename=f"{log_dir}/monitor.csv")  # Logs training metrics

# === 3. Action noise ===
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# === 4. Evaluation callback ===
eval_callback = EvalCallback(
    env,
    best_model_save_path=f"{model_dir}/best_model/",
    log_path=log_dir,
    eval_freq=5000,              # Evaluate every 5000 steps
    deterministic=True,
    render=False
)

# === 5. Define TD3 model ===
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

# === 6. Train model ===
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME, callback=eval_callback)

# === 7. Save final model ===
model.save(f"{model_dir}/td3_pricing_model")
print(f"✅ TD3 model saved to {model_dir}/td3_pricing_model")