# evaluate_dqn_simple.py
from stable_baselines3 import DQN
from env.old.ecommerce_env_simple2 import AdvancedPricingEnv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Load environment and model ===
env = AdvancedPricingEnv()
model = DQN.load("models/dqn/DQN_run_advanced/dqn_pricing_model")

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
df.to_csv("data/generated/eval_output_dqn_advanced.csv", index=False)
print("✅ Evaluation saved to data/generated/eval_output_dqn_advanced.csv")

# === 4. Plot ===
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['quantity_sold'])
plt.title("📉 Demand vs Price")
plt.xlabel("Price")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("analysis/advanced/plot_dqn_demand_vs_price.png")
plt.close()

plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['reward'])
plt.title("📈 Reward over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("analysis/advanced/plot_dqn_reward_over_time.png")
plt.close()

print("📊 Graphs saved to data/generated/")
