from stable_baselines3 import TD3
from env.ecommerce_env import EcommercePricingEnv
import pandas as pd
import os

# === RL AGENT (TD3) ===
env = EcommercePricingEnv()
model = TD3.load("models/td3/TD3_run1/td3_pricing_model")

obs = env.reset()[0]
done = False
total_reward = 0
steps = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    steps.append(info)

print(f"🤖 TD3 Agent Total Reward: {total_reward:.2f}")

# === Export actions and results ===
os.makedirs("data/generated", exist_ok=True)
df = pd.DataFrame(steps)
df.to_csv("data/generated/eval_output_td3.csv", index=False)
print("✅ Evaluation log saved to data/generated/eval_output_td3.csv")