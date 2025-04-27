import matplotlib.pyplot as plt
import pandas as pd
import os

# Load evaluation output
df = pd.read_csv("data/generated/eval_output_dqn_real2.csv")

# 2. Load the translation + super category mapping
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")

# 3. Merge based on translated name
df = df.merge(
    translation[['translated_name', 'main_category']],
    how='left',
    left_on='product_category_name',
    right_on='translated_name'
)

df.head()

# Identify top 6 categories by total sales
top_categories = (df.groupby('main_category')['quantity_sold']
                    .sum()
                    .sort_values(ascending=False)
                    .head(6)
                    .index)

# Make sure 'step' is treated as week
df['week'] = df['step']

# Create folder for individual plots if needed
os.makedirs("analysis/real2/category_plots", exist_ok=True)

# Generate plot for each top category
for category in top_categories:
    df_cat = df[df['main_category'] == category].copy()

    agg = df_cat.groupby('week').agg({
        'new_price': 'mean',
        'total_demand': 'sum',
        'quantity_sold': 'sum'
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Savaitė")
    ax1.set_ylabel("Kaina (€)", color='tab:blue')
    ax1.plot(agg['week'], agg['new_price'], color='tab:blue', label='Kaina')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("Vnt", color='tab:orange')
    ax2.plot(agg['week'], agg['total_demand'], color='tab:orange', linestyle='--', label='Paklausa')
    ax2.plot(agg['week'], agg['quantity_sold'], color='tab:green', linestyle='-', label='Pardavimai')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f"Kategorija: {category} - Kainos, paklausos ir pardavimų dinamika")
    plt.tight_layout()
    plt.savefig(f"analysis/real2/category_plots/weekly_price_demand_sold_{category}.png")
    plt.close()

print("✅ Grafikai sukurti kiekvienai kategorijai!")