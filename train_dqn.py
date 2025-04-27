from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env.old.ecommerce_env import EcommercePricingEnv
import os
import pandas as pd

# Custom callback to log rewards
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("dones"):
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    ep_info = self.locals["infos"][idx]
                    if "episode" in ep_info:
                        self.episode_rewards.append(ep_info["episode"]["r"])
        return True

    def _on_training_end(self) -> None:
        # Save rewards to CSV
        pd.DataFrame(self.episode_rewards, columns=["reward"]).to_csv(self.log_path, index=False)
        print(f"📦 Reward log saved to {self.log_path}")

# Prepare environment
env = DummyVecEnv([lambda: Monitor(EcommercePricingEnv())])  # Works fine!

# Save model and logs
RUN_NAME = "DQN_run1"
model_dir = f"models/dqn/{RUN_NAME}"
os.makedirs(model_dir, exist_ok=True)

log_path = f"{model_dir}/rewards.csv"
callback = RewardLoggerCallback(log_path)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS, tb_log_name=RUN_NAME, callback=callback)

model.save(f"{model_dir}/dqn_pricing_model")