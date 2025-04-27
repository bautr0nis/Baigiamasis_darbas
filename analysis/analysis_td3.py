import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load evaluation output ===
df = pd.read_csv("../data/generated/eval_output_td3_analytical.csv")
os.makedirs("../data/generated/analysis", exist_ok=True)

# === 1. Reward per episode ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['reward'])
plt.title("📈 Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("../data/generated/analysis/reward_over_time.png")
plt.close()

# === 2. Price over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['new_price'])
plt.title("💶 Price Over Time")
plt.xlabel("Step")
plt.ylabel("Price (€)")
plt.tight_layout()
plt.savefig("../data/generated/analysis/price_over_time.png")
plt.close()

# === 3. Profit vs Price ===
df['profit'] = (df['new_price'] - df['cost']) * df['quantity_sold']
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['profit'])
plt.title("💰 Profit vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Profit (€)")
plt.tight_layout()
plt.savefig("../data/generated/analysis/profit_vs_price.png")
plt.close()

# === 4. Price Volatility ===
price_std = df['new_price'].std()
print(f"📊 Price volatility (std dev): {price_std:.2f}€")

# === 5. Demand Sensitivity ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['quantity_sold'])
plt.title("📉 Demand vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("../data/generated/analysis/demand_vs_price.png")
plt.close()

# === 6. Summary statistics ===
avg_reward = df['reward'].mean()
total_profit = df['profit'].sum()
print(f"✅ Average reward: {avg_reward:.2f}")
print(f"💸 Total profit: {total_profit:.2f}€")

# === 6. Kainų keitimo analizė ===
price_changes = df['new_price'] - df['price']
increase_pct = (price_changes > 1e-2).mean() * 100
decrease_pct = (price_changes < -1e-2).mean() * 100
neutral_pct = 100 - increase_pct - decrease_pct

# Spausdinam
print("\n📊 Kainos pokyčio pasiskirstymas:")
print(f"⬆️ Kėlė kainą: {increase_pct:.2f}%")
print(f"⬇️ Mažino kainą: {decrease_pct:.2f}%")
print(f"➖ Nepakeitė: {neutral_pct:.2f}%")

# Pie chart
labels = ['Kaina pakelta', 'Kaina sumažinta', 'Kaina nepakitusi']
sizes = [increase_pct, decrease_pct, neutral_pct]
colors = ['#66b3ff', '#ff9999', '#dddddd']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("🎯 Kainų pokyčių pasiskirstymas")
plt.tight_layout()
plt.savefig("../data/generated/plot_price_change_distribution.png")
plt.close()

# Optional: Save summary
summary = {
    "avg_reward": avg_reward,
    "total_profit": total_profit,
    "price_volatility": price_std,
    "max_price": df['new_price'].max(),
    "min_price": df['new_price'].min(),
    "mean_quantity": df['quantity_sold'].mean()
}
pd.DataFrame([summary]).to_csv("../data/generated/analysis/summary_metrics.csv", index=False)
print("📄 Summary saved to summary_metrics.csv")