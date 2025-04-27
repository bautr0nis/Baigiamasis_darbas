import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# === 0. Setup output folder ===
os.makedirs("analysis", exist_ok=True)

# === 1. Load evaluation output ===
df = pd.read_csv("../data/generated/eval_output_td3.csv")

# === 2. Drop NaNs ===
df = df.dropna(subset=['price', 'quantity_sold'])

# === 3. Check if quantity_sold is not constant ===
if df['quantity_sold'].nunique() > 1:

    # === 4. Koreliacija ===
    corr, p_value = pearsonr(df['price'], df['quantity_sold'])
    print(f"📊 Koreliacija tarp kainos ir pardavimų: r = {corr:.3f}, p = {p_value:.4f}")

    # === 5. Vizualizacija ===
    plt.figure(figsize=(8, 5))
    sns.regplot(x='price', y='quantity_sold', data=df, scatter_kws={"alpha": 0.4})
    plt.title("📉 Price vs Quantity Sold")
    plt.xlabel("Price")
    plt.ylabel("Quantity Sold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_vs_quantity_corr.png")
    plt.show()

else:
    print("⚠️ Visi quantity_sold yra vienodi – negalima apskaičiuoti koreliacijos.")