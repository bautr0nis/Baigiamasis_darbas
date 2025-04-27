import matplotlib.pyplot as plt
import pandas as pd
import os

# Load evaluation output
df = pd.read_csv("data/generated/eval_output_dqn_real.csv")

# Identify top 6 categories by total sales
top_categories = (df.groupby('product_category_name')['quantity_sold']
                    .sum()
                    .sort_values(ascending=False)
                    .head(6)
                    .index)

# Make sure 'step' is treated as week
df['week'] = df['step']

# Create folder for individual plots if needed
os.makedirs("analysis/real/category_plots", exist_ok=True)

# Moving average window (pvz., 3 savaitės)
window_size = 3

# Generate plot for each top category
for category in top_categories:
    df_cat = df[df['product_category_name'] == category].copy()

    agg = df_cat.groupby('week').agg({
        'new_price': 'mean',
        'total_demand': 'sum',
        'quantity_sold': 'sum'
    }).reset_index()

    # Taikome moving average
    agg['new_price_smooth'] = agg['new_price'].rolling(window=window_size, min_periods=1).mean()
    agg['total_demand_smooth'] = agg['total_demand'].rolling(window=window_size, min_periods=1).mean()
    agg['quantity_sold_smooth'] = agg['quantity_sold'].rolling(window=window_size, min_periods=1).mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Savaitė")
    ax1.set_ylabel("Kaina (€)", color='tab:blue')
    ax1.plot(agg['week'], agg['new_price_smooth'], color='tab:blue', label='Kaina (smooth)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Vnt", color='tab:orange')
    ax2.plot(agg['week'], agg['total_demand_smooth'], color='tab:orange', linestyle='--', label='Paklausa (smooth)')
    ax2.plot(agg['week'], agg['quantity_sold_smooth'], color='tab:green', linestyle='-', label='Pardavimai (smooth)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f"Kategorija: {category} - Kainos, paklausos ir pardavimų dinamika (su moving average)")
    plt.tight_layout()
    plt.savefig(f"analysis/real/category_plots/weekly_price_demand_sold_{category}_smooth.png")
    plt.close()

print("✅ Grafikai sukurti kiekvienai kategorijai su moving average!")