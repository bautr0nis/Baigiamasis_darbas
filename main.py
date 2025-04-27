# analyze_dqn_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Load evaluation output ===
df = pd.read_csv("data/generated/eval_output_dqn_simple.csv")
os.makedirs("data/generated/analysis", exist_ok=True)

# === 2. Reward over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['reward'])
plt.title("📈 DQN Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward (€)")
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_reward_over_time.png")
plt.close()

# === 3. Price over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['new_price'])
plt.title("💶 Price Over Time")
plt.xlabel("Step")
plt.ylabel("New Price (€)")
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_price_over_time.png")
plt.close()

# === 4. Profit vs Price ===
df['profit'] = (df['new_price'] - df['cost']) * df['quantity_sold']
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['profit'])
plt.title("💰 Profit vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Profit (€)")
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_profit_vs_price.png")
plt.close()

# === 5. Demand vs Price ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['quantity_sold'])
plt.title("📉 Demand vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_demand_vs_price.png")
plt.close()

# === 6. Price change stats ===
df['price_change'] = df['new_price'] - df['price']
increase_pct = (df['price_change'] > 1e-2).mean() * 100
decrease_pct = (df['price_change'] < -1e-2).mean() * 100
neutral_pct = 100 - increase_pct - decrease_pct

print("\n📊 Kainos pokyčio pasiskirstymas:")
print(f"⬆️ Kėlė kainą: {increase_pct:.2f}%")
print(f"⬇️ Mažino kainą: {decrease_pct:.2f}%")
print(f"➖ Nepakeitė: {neutral_pct:.2f}%")

# === 7. Pie chart ===
labels = ['Kaina pakelta', 'Kaina sumažinta', 'Kaina nepakitusi']
sizes = [increase_pct, decrease_pct, neutral_pct]
colors = ['#66b3ff', '#ff9999', '#dddddd']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("🎯 Kainų pokyčių pasiskirstymas")
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_price_change_distribution.png")
plt.close()

# === 8. Summary metrics ===
price_std = df['new_price'].std()
avg_reward = df['reward'].mean()
total_profit = df['profit'].sum()
summary = {
    "avg_reward": avg_reward,
    "total_profit": total_profit,
    "price_volatility": price_std,
    "max_price": df['new_price'].max(),
    "min_price": df['new_price'].min(),
    "mean_quantity": df['quantity_sold'].mean()
}
pd.DataFrame([summary]).to_csv("data/generated/analysis/dqn_summary_metrics.csv", index=False)

print("\n✅ Išsaugota analizė aplanke: data/generated/analysis/")


df_mon = pd.read_csv("logs/dqn/monitor.csv", skiprows=1)  # pirmoje eilutėje yra metadata
df_mon['rolling_reward'] = df_mon['r'].rolling(window=50).mean()

plt.figure(figsize=(12, 5))
sns.lineplot(data=df_mon, x=df_mon.index, y='rolling_reward')
plt.title("📊 DQN Training Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Rolling Avg Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/generated/analysis/dqn_training_reward_full.png")
plt.close()


rl_total_profit = df['profit'].sum()

# === Baseline: naudok originalią price kaip new_price ===
df['baseline_profit'] = (df['price'] - df['cost']) * df['quantity_sold']
baseline_profit = df['baseline_profit'].sum()

profit_diff = rl_total_profit - baseline_profit
profit_diff_pct = (profit_diff / baseline_profit) * 100

print("\n📊 Kainos skirtumas:")
print(f"⬆️ Kainos skirtumas: {profit_diff}")
print(f"⬆️ Kainos skirtumas proc: {profit_diff_pct:.2f}%")