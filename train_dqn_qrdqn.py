# train_dqn_qrdqn.py
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env.ecommerce_env import EcommercePricingEnv
from gymnasium.wrappers import RecordEpisodeStatistics
import os
import pandas as pd

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "episode" in info and info["episode"] is not None:
                    self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        pd.DataFrame(self.episode_rewards, columns=["reward"]).to_csv(self.log_path, index=False)
        print(f"📦 Reward log saved to {self.log_path}")

# Prepare environment
env = DummyVecEnv([lambda: RecordEpisodeStatistics(EcommercePricingEnv())])

# Setup directories
RUN_NAME = "QRDQN_run1"
model_dir = f"models/qrdqn/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)
log_path = f"{model_dir}/rewards.csv"
callback = RewardLoggerCallback(log_path)

# Define model
model = QRDQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    train_freq=10,
    target_update_interval=1000,
    exploration_fraction=0.15,
    exploration_final_eps=0.05,
    tensorboard_log="./tensorboard/"
)

# Train
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME, callback=callback)

# Save
model.save(f"{model_dir}/qrdqn_pricing_model")
print(f"✅ QRDQN model saved at {model_dir}/qrdqn_pricing_model")
