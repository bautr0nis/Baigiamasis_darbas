import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Load evaluation output ===
df = pd.read_csv("data/generated/eval_output_dqn_advanced.csv")
os.makedirs("data/generated/analysis", exist_ok=True)

# === 2. Reward over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['reward'])
plt.title("DQN Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward (€)")
plt.tight_layout()
plt.savefig("analysis/advanced/dqn_reward_over_time.png")
plt.close()

# === 3. Price over time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['new_price'])
plt.title("Price Over Time")
plt.xlabel("Step")
plt.ylabel("New Price (€)")
plt.tight_layout()
plt.savefig("analysis/advanced/dqn_price_over_time.png")
plt.close()

# === 4. Profit vs Price ===
df['profit'] = (df['new_price'] - df['cost']) * df['quantity_sold']
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['profit'])
plt.title("Profit vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Profit (€)")
plt.tight_layout()
plt.savefig("analysis/advanced/dqn_profit_vs_price.png")
plt.close()

# === 5. Demand vs Price ===
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['new_price'], y=df['quantity_sold'])
plt.title("Demand vs Price")
plt.xlabel("Price (€)")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("analysis/advanced/dqn_demand_vs_price.png")
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
plt.savefig("analysis/advanced/dqn_price_change_distribution.png")
plt.close()

# === 8. Demand level before action vs action type ===
plt.figure(figsize=(8, 5))
sns.boxplot(x='action_type', y='quantity_sold', data=df)
plt.title("Paklausa prieš veiksmą (kainos keitimas)")
plt.tight_layout()
plt.savefig("analysis/advanced/demand_vs_action_type.png")
plt.close()

# === 9. Price change frequency by category ===
plt.figure(figsize=(10, 6))
category_action = df.groupby(['product_category_name', 'action_type']).size().unstack().fillna(0)
category_action_pct = category_action.div(category_action.sum(axis=1), axis=0) * 100
category_action_pct.plot(kind='bar', stacked=True)
plt.title("Kainos pokyčiai pagal kategoriją")
plt.ylabel("% veiksmų")
plt.tight_layout()
plt.savefig("analysis/advanced/price_change_by_category.png")
plt.close()

# === 10. Elastingumas vs Price Change ===
plt.figure(figsize=(8, 5))
sns.boxplot(x='action_type', y='price_elasticity', data=df)
plt.title("Elastingumas pagal veiksmą")
plt.tight_layout()
plt.savefig("analysis/advanced/elasticity_vs_action.png")
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

# === 14. Price increase violations > 10%
df['price_change_pct'] = df['price_change'] / df['price']
df['price_increase_violation'] = df['price_change_pct'] > 0.10

violation_count = df['price_increase_violation'].sum()
violation_pct = df['price_increase_violation'].mean() * 100

print(f"\n🚨 Kainos pokyčių > +10% pažeidimų skaičius: {violation_count} ({violation_pct:.2f}%)")

plt.figure(figsize=(6, 5))
sns.countplot(x='price_increase_violation', data=df)
plt.title("🚨 Kainų >10% padidėjimų skaičius")
plt.xlabel("Pažeidimas")
plt.ylabel("Epizodų skaičius")
plt.xticks([0, 1], ['Ne', 'Taip'])
plt.tight_layout()
plt.savefig("analysis/advanced/price_violation_count.png")
plt.close()

# === 15. Demand > Stock (over-demand) impact
df['over_demand'] = df['total_demand'] > df['stock']

plt.figure(figsize=(6, 5))
sns.countplot(x='over_demand', data=df)
plt.title("📦 Ar paklausa viršijo sandėlį?")
plt.xlabel("Paklausa > Sandėlis")
plt.ylabel("Skaičius")
plt.xticks([0, 1], ['Ne', 'Taip'])
plt.tight_layout()
plt.savefig("analysis/advanced/over_demand_cases.png")
plt.close()

over_demand_profit = df.groupby('over_demand')['profit'].mean()
print("\n📦 Vidutinis pelnas kai paklausa > sandėlis:")
print(over_demand_profit)

# === 16. Rolling reward + price change over time
df['rolling_reward'] = df['reward'].rolling(10).mean()
df['rolling_price'] = df['new_price'].rolling(10).mean()

plt.figure(figsize=(12, 6))
sns.lineplot(x=df['step'], y=df['rolling_reward'], label="Reward (avg)")
sns.lineplot(x=df['step'], y=df['rolling_price'], label="Price (avg)")
plt.title("⏱️ Rolling Reward ir Kainos Pokyčiai per Laiką")
plt.xlabel("Step")
plt.ylabel("Reikšmė")
plt.legend()
plt.tight_layout()
plt.savefig("analysis/advanced/rolling_reward_price.png")
plt.close()

## elastingumas
category_elasticity = df.groupby("product_category_name")["price_elasticity"].mean().sort_values()
plt.figure(figsize=(10, 6))
sns.barplot(x=category_elasticity.values, y=category_elasticity.index)
plt.title("Vidutinis elastingumas pagal kategoriją")
plt.xlabel("Price Elasticity")
plt.tight_layout()
plt.savefig("analysis/advanced/category_elasticity.png")
plt.close()

# === 14. Price change action distribution ===
df['price_change_pct'] = (df['action'] - 4) * 5  # % pokytis: nuo -20% iki +20% (žingsnis 5%)

# Skaičiuojam dažnius
action_dist = df['price_change_pct'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(x=action_dist.index, y=action_dist.values, palette="coolwarm")
plt.xlabel("Kainos pokytis (%)")
plt.ylabel("Veiksmų skaičius")
plt.title("📊 Kainų keitimo veiksmų pasiskirstymas (RL sprendimai)")
plt.tight_layout()
plt.savefig("analysis/advanced/price_change_action_distribution.png")
plt.close()


#VIzualuas paaiskinamumas per laika
# 📈 Savaitinis grafikas: kaina, paklausa ir pardavimai
weekly = df.copy()
weekly['week'] = weekly['step'] // 7

agg = weekly.groupby('week').agg({
    'new_price': 'mean',
    'total_demand': 'sum',
    'quantity_sold': 'sum'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot price
ax1.set_xlabel("Savaitė")
ax1.set_ylabel("Kaina (€)", color='tab:blue')
ax1.plot(agg['week'], agg['new_price'], color='tab:blue', label='Kaina')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Add second Y axis for demand and sales
ax2 = ax1.twinx()
ax2.set_ylabel("Vnt", color='tab:orange')
ax2.plot(agg['week'], agg['total_demand'], color='tab:orange', linestyle='--', label='Paklausa')
ax2.plot(agg['week'], agg['quantity_sold'], color='tab:green', linestyle='-', label='Pardavimai')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title("Kainos, paklausos ir pardavimų dinamika per savaites")
plt.tight_layout()
plt.savefig("analysis/advanced/weekly_price_demand_sold.png")
plt.close()


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
plt.savefig("analysis/advanced/full_dqn_training_reward.png")
plt.close()
