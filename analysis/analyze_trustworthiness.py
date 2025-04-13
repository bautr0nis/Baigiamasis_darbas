# analysis_trustworthiness.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 0. Setup ===
os.makedirs("analysis", exist_ok=True)
df = pd.read_csv("data/generated/eval_output.csv")

# === 1. Price over Time / Episode ===
plt.figure(figsize=(12, 4))
sns.lineplot(data=df, x='step', y='new_price', color='red')
plt.title("🔁 Price over Time")
plt.xlabel("Time step")
plt.ylabel("Price ($)")
plt.tight_layout()
plt.savefig("analysis/price_over_time.png")
plt.close()

# === 2. Profit vs Price Curve (Simulated) ===
if {'price', 'cost', 'freight_value', 'quantity_sold'}.issubset(df.columns):
    df['profit'] = (df['new_price'] - df['cost'] - df['freight_value']) * df['quantity_sold']
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=df.sort_values("new_price"), x="new_price", y="profit")
    plt.title("💰 Profit vs Price")
    plt.xlabel("Price")
    plt.ylabel("Profit")
    plt.tight_layout()
    plt.savefig("analysis/profit_vs_price.png")
    plt.close()

# === 3. RL agent return per episode ===
# Calculate cumulative return with rolling average + std
df['episode'] = df['step'] // 100  # Assuming 100 steps per episode
returns = df.groupby('episode')['reward'].sum().reset_index()
returns['rolling_mean'] = returns['reward'].rolling(window=10).mean()
returns['rolling_std'] = returns['reward'].rolling(window=10).std()

plt.figure(figsize=(12, 4))
plt.plot(returns['episode'], returns['rolling_mean'], label='Rolling Return (mean)')
plt.fill_between(returns['episode'],
                 returns['rolling_mean'] - returns['rolling_std'],
                 returns['rolling_mean'] + returns['rolling_std'],
                 color='blue', alpha=0.2, label='±1 std')
plt.title("📈 Return per Episode with Uncertainty")
plt.xlabel("Episode")
plt.ylabel("Return ($)")
plt.legend()
plt.tight_layout()
plt.savefig("analysis/return_per_episode.png")
plt.close()

# === 4. Price Change Frequency ===
df['price_change_flag'] = df['price'] != df['new_price']
price_change_counts = df.groupby('episode')['price_change_flag'].mean().reset_index()

plt.figure(figsize=(12, 4))
sns.lineplot(data=price_change_counts, x='episode', y='price_change_flag')
plt.title("📊 Price Change Frequency per Episode")
plt.xlabel("Episode")
plt.ylabel("Change Frequency")
plt.tight_layout()
plt.savefig("analysis/price_change_frequency.png")
plt.close()

print("✅ Trustworthy RL analysis graphs saved to /analysis/")