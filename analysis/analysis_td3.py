import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === 1. Load evaluation results ===
os.makedirs("analysis", exist_ok=True)
df = pd.read_csv("data/generated/eval_output_td3.csv")

# === 2. Price over Time ===
plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['new_price'])
plt.title("📈 Price Over Time")
plt.xlabel("Step")
plt.ylabel("New Price")
plt.tight_layout()
plt.savefig("analysis/td3_price_over_time.png")
plt.close()

# === 3. Profit vs Price Change ===
if {'price', 'new_price', 'cost', 'freight_value', 'quantity_sold'}.issubset(df.columns):
    df['price_change_%'] = df['new_price'] / df['price'] - 1
    df['profit'] = (df['new_price'] - df['cost'] - df['freight_value']) * df['quantity_sold']
    profit_bins = pd.cut(df['price_change_%'], bins=[-1, -0.2, -0.1, 0, 0.1, 0.2, 1])
    profit_by_change = df.groupby(profit_bins)['profit'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=profit_by_change, x='price_change_%', y='profit')
    plt.title("💰 Average Profit vs Price Change")
    plt.xlabel("Price Change (%)")
    plt.ylabel("Average Profit")
    plt.tight_layout()
    plt.savefig("analysis/td3_profit_vs_price_change.png")
    plt.close()

# === 4. Return per Episode ===
EPISODE_LENGTH = 100
returns = df.groupby(df['step'] // EPISODE_LENGTH)['reward'].sum().reset_index(name='return')

plt.figure(figsize=(10, 5))
sns.lineplot(data=returns, x='step', y='return')
plt.title("📊 Return per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig("analysis/td3_return_per_episode.png")
plt.close()

# === 5. Price Change Frequency ===
df['price_change'] = (df['new_price'] - df['price']).abs()
df['rolling_price_change'] = df['price_change'].rolling(window=10).mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=df['step'], y=df['rolling_price_change'])
plt.title("🔁 Rolling Price Change Frequency")
plt.xlabel("Step")
plt.ylabel("|Price Change| (Rolling Mean)")
plt.tight_layout()
plt.savefig("analysis/td3_price_change_freq.png")
plt.close()

print("✅ TD3 analysis completed and saved to /analysis")
