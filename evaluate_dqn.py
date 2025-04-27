import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from env.old.ecommerce_env_analytical import EcommercePricingEnv
import os

# === 1. Load env & model ===
env = EcommercePricingEnv(data_path="data/synthetic_olist_data.csv", verbose=False)
model = DQN.load("models/dqn/DQN_run_discrete/dqn_pricing_model")

# === 2. Run evaluation ===
obs = env.reset()[0]
done = False
total_reward = 0
steps = []
actions = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    steps.append(info)
    actions.append(action)  # Discrete action

    print(f"[Step {info['step']}] 💵 Action: {action}, Price: {info['price']:.2f} → {info['new_price']:.2f}, Qty: {info['quantity_sold']}, Reward: {reward:.2f}")

# === 3. Save evaluation results ===
df = pd.DataFrame(steps)
df["action"] = actions
df["price_change"] = df["new_price"] - df["price"]

os.makedirs("data/generated", exist_ok=True)
df.to_csv("data/generated/eval_output_dqn_analytical.csv", index=False)
print(f"✅ Evaluation results saved to data/generated/eval_output_dqn_analytical.csv")

# === 4. Visualizations ===
sns.set(style="whitegrid")

# 📊 Price Change Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["price_change"], bins=30)
plt.title("📉 Kainos pokyčių pasiskirstymas")
plt.xlabel("Nauja kaina - Sena kaina")
plt.tight_layout()
plt.savefig("data/generated/plot_price_change_histogram_dqn.png")
plt.close()

# 📊 Action Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["action"], bins=30)
plt.title("🎮 Veiksmų (Action) pasiskirstymas")
plt.xlabel("Veiksmo reikšmė")
plt.tight_layout()
plt.savefig("data/generated/plot_action_distribution_dqn.png")
plt.close()

print("📊 Papildomi grafikai išsaugoti į data/generated/")