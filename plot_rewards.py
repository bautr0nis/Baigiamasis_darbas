import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from env.old.ecommerce_env_analytical import EcommercePricingEnv
from stable_baselines3 import TD3

sns.set(style="whitegrid")

# === 1. TD3 Training Reward Plot ===
monitor_path = "logs/td3/monitor.csv"
df_monitor = pd.read_csv(monitor_path, skiprows=1)
df_monitor['rolling_reward'] = df_monitor['r'].rolling(window=50).mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=df_monitor.index, y='rolling_reward', data=df_monitor)
plt.title("📊 TD3 Training Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Reward (Rolling Avg)")
plt.grid(True)
plt.tight_layout()
os.makedirs("data/generated", exist_ok=True)
plt.savefig("data/generated/plot_training_reward.png")
plt.close()

print("✅ TD3 training reward plot saved")

# === 2. Evaluate TD3 Agent ===
env_td3 = EcommercePricingEnv(data_path="data/synthetic_olist_data.csv", verbose=False)
model = TD3.load("models/td3/TD3_run_analytical/td3_pricing_model")

obs = env_td3.reset()[0]
done = False
total_td3_reward = 0
steps_td3 = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env_td3.step(action)
    total_td3_reward += reward
    steps_td3.append(info)

df_td3 = pd.DataFrame(steps_td3)
df_td3.to_csv("data/generated/eval_output_td3.csv", index=False)

# === 3. Evaluate Baseline (Fixed price strategy) ===
env_base = EcommercePricingEnv(data_path="data/synthetic_olist_data.csv", verbose=False)
obs = env_base.reset()[0]
done = False
total_baseline_reward = 0
steps_base = []

while not done:
    action = [0.0]  # always use base price
    obs, reward, done, _, info = env_base.step(action)
    total_baseline_reward += reward
    steps_base.append(info)

df_base = pd.DataFrame(steps_base)
df_base.to_csv("data/generated/eval_output_baseline.csv", index=False)

# === 4. Comparison Metrics ===
reward_diff = total_td3_reward - total_baseline_reward
percent_improvement = (reward_diff / total_baseline_reward) * 100

print("\n🎯 Performance Comparison")
print(f"Baseline total reward:   €{total_baseline_reward:,.2f}")
print(f"TD3 agent total reward:  €{total_td3_reward:,.2f}")
print(f"➕ Absolute gain:         €{reward_diff:,.2f}")
print(f"📈 % improvement:        {percent_improvement:.2f}%")

# === 5. Plot Comparison: Reward Over Time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df_td3['step'], y=df_td3['reward'], label="TD3 Agent")
sns.lineplot(x=df_base['step'], y=df_base['reward'], label="Baseline (Fixed Price)")
plt.title("📈 Reward Over Time: TD3 vs Baseline")
plt.xlabel("Step")
plt.ylabel("Reward (€)")
plt.legend()
plt.tight_layout()
plt.savefig("data/generated/plot_compare_reward_over_time.png")
plt.close()

print("📊 All evaluation plots and data saved to data/generated/")