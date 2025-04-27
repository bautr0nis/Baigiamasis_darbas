# evaluate_dqn_simple.py
from stable_baselines3 import DQN
from env.ecommerce_env_real2 import AdvancedPricingEnv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Load environment and model ===
env = AdvancedPricingEnv()
model = DQN.load("models/dqn/DQN_run_real2/dqn_pricing_model2")

# === 2. Evaluation ===
obs = env.reset()[0]
done = False
total_reward = 0
steps = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    info["action"] = int(action)  # <- PRIDĖTA čia!
    total_reward += reward
    steps.append(info)
    print(f"[Step {info['step']}] 💵 Action: {action}, Price: {info['price']:.2f} → {info['new_price']:.2f}, Qty: {info['quantity_sold']}, Reward: {reward:.2f}")

print(f"\n🤖 DQN Total Reward: {total_reward:.2f}")

# === 3. Save output ===
os.makedirs("data/generated", exist_ok=True)
df = pd.DataFrame(steps)
df.to_csv("data/generated/eval_output_dqn_real2.csv", index=False)
print("✅ Evaluation saved to data/generated/eval_output_dqn_real2.csv")


print("📊 Graphs saved to data/generated/")
