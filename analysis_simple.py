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
plt.title("DQN Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward (€)")
plt.tight_layout()
plt.savefig("analysis/simple/dqn_reward_over_time.png")
plt.close()

# === 3. Price over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['new_price'])
plt.title("Price Over Time")
plt.xlabel("Step")
plt.ylabel("New Price (€)")
plt.tight_layout()
plt.savefig("analysis/simple/dqn_price_over_time.png")
plt.close()

# === 4. Profit vs Price ===
df['profit'] = (df['new_price'] - df['cost']) * df['quantity_sold']
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['profit'])
plt.title("Profit vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Profit (€)")
plt.tight_layout()
plt.savefig("analysis/simple/dqn_profit_vs_price.png")
plt.close()

# === 5. Demand vs Price ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['quantity_sold'])
plt.title("Demand vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("analysis/simple/dqn_demand_vs_price.png")
plt.close()

# === 6. Price change stats ===
df['price_change'] = df['new_price'] - df['price']
df['action_type'] = df['price_change'].apply(lambda x: 'increase' if x > 1e-2 else ('decrease' if x < -1e-2 else 'no_change'))
increase_pct = (df['action_type'] == 'increase').mean() * 100
decrease_pct = (df['action_type'] == 'decrease').mean() * 100
neutral_pct = 100 - increase_pct - decrease_pct

print("Kainos pokyčio pasiskirstymas:")
print(f"⬆️ Kėlė kainą: {increase_pct:.2f}%")
print(f"🔽️ Mažino kainą: {decrease_pct:.2f}%")
print(f"➖ Nepakeitė: {neutral_pct:.2f}%")

# === 7. Pie chart ===
labels = ['Kaina pakelta', 'Kaina sumažinta', 'Kaina nepakitusi']
sizes = [increase_pct, decrease_pct, neutral_pct]
colors = ['#66b3ff', '#ff9999', '#dddddd']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title("Kainų pokyčių pasiskirstymas")
plt.tight_layout()
plt.savefig("analysis/simple/dqn_price_change_distribution.png")
plt.close()

# === 8. Demand level before action vs action type ===
plt.figure(figsize=(8, 5))
sns.boxplot(x='action_type', y='quantity_sold', data=df)
plt.title("Paklausa prieš veiksmą (kainos keitimas)")
plt.tight_layout()
plt.savefig("analysis/simple/demand_vs_action_type.png")
plt.close()

# === 9. Price change frequency by category ===
plt.figure(figsize=(10, 6))
category_action = df.groupby(['product_category_name', 'action_type']).size().unstack().fillna(0)
category_action_pct = category_action.div(category_action.sum(axis=1), axis=0) * 100
category_action_pct.plot(kind='bar', stacked=True)
plt.title("Kainos pokyčiai pagal kategoriją")
plt.ylabel("% veiksmų")
plt.tight_layout()
plt.savefig("analysis/simple/price_change_by_category.png")
plt.close()

# === 10. Elastingumas vs Price Change ===
plt.figure(figsize=(8, 5))
sns.boxplot(x='action_type', y='price_elasticity', data=df)
plt.title("Elastingumas pagal veiksmą")
plt.tight_layout()
plt.savefig("analysis/simple/elasticity_vs_action.png")
plt.close()

# === 11. Policy rationality check: Low demand -> reduce? ===
df['low_demand'] = df['quantity_sold'] < 5
policy_match = df[df['low_demand']]['action_type'].value_counts(normalize=True) * 100
print("Ar RL elgėsi verslo logikos rėmuose kai paklausa maža?")
print(policy_match)


# === 12. RL vs Baseline Profit ===
df['baseline_profit'] = (df['price'] - df['cost']) * df['quantity_sold']
rl_total_profit = df['profit'].sum()
baseline_profit = df['baseline_profit'].sum()
profit_diff = rl_total_profit - baseline_profit
profit_diff_pct = (profit_diff / baseline_profit) * 100

print(f"RL Total Profit: {rl_total_profit:.2f}€")
print(f"Baseline Profit: {baseline_profit:.2f}€")
print(f"Skirtumas: {profit_diff:.2f}€ ({profit_diff_pct:.2f}%)")

# === 13. Summary metrics ===
summary = {
    "avg_reward": df['reward'].mean(),
    "total_profit_rl": rl_total_profit,
    "total_profit_baseline": baseline_profit,
    "profit_difference": profit_diff,
    "profit_difference_pct": profit_diff_pct,
    "price_volatility": df['new_price'].std(),
    "max_price": df['new_price'].max(),
    "min_price": df['new_price'].min(),
    "mean_quantity": df['quantity_sold'].mean()
}
pd.DataFrame([summary]).to_csv("data/generated/analysis/dqn_summary_metrics.csv", index=False)

print("\n\u2705 Išsaugota analizė aplanke: data/generated/analysis/")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# === FULL CURVE ===
# === 1. Load monitor log ===
log_path = "logs/dqn/monitor.csv"  # kelias gali skirtis pagal tavo projektą
df = pd.read_csv(log_path, skiprows=1)  # pirmoji eilutė – komentaras

# === 2. Rolling average reward ===
df['rolling_reward'] = df['r'].rolling(window=50).mean()

# === 3. Plot ===
plt.figure(figsize=(12, 5))
sns.lineplot(x=df.index, y='rolling_reward', data=df)
plt.title("📈 DQN Training Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Rolling Avg Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("analysis/simple/full_dqn_training_reward.png")
plt.close()

#