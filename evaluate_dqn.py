from stable_baselines3 import DQN
from env.ecommerce_env import EcommercePricingEnv
import pandas as pd

# === RL AGENT ===
env = EcommercePricingEnv()
model = DQN.load("models/dqn/DQN_run1/dqn_pricing_model")

obs = env.reset()[0]
done = False
total_reward = 0
steps = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    steps.append(info)  # store step info (includes price, quantity, etc.)

print(f"🤖 RL Agent Total Reward: {total_reward:.2f}")

# === Export actions and results for analysis ===
df = pd.DataFrame(steps)
df.to_csv("data/generated/eval_output.csv", index=False)
print("✅ Evaluation log saved to data/generated/eval_output.csv")