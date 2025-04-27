# optimal_price_simulation_adjusted.py

import numpy as np
import matplotlib.pyplot as plt

# === Parametrai ===
base_price = 100
cost = 60
base_demand = 100  # 🚀 padidinta pradinė paklausa
price_elasticity = -4.0  # 📉 jautresnis kainai (didesnis neigiamas skaičius)

# === Simuliacija ===
prices = np.linspace(50, 150, 100)
demands = base_demand * (prices / base_price) ** price_elasticity
demands = np.clip(np.round(demands), 0, None)  # 🔢 suapvalinta ir be neigiamų reikšmių
profits = (prices - cost) * demands

# === Optimali reikšmė ===
optimal_idx = np.argmax(profits)
optimal_price = prices[optimal_idx]
optimal_profit = profits[optimal_idx]
optimal_demand = demands[optimal_idx]

# === Vizualizacija ===
plt.figure(figsize=(10, 5))
plt.plot(prices, profits, label="Pelnas")
plt.plot(prices, demands, '--', label="Paklausa")
plt.axvline(optimal_price, color='r', linestyle=':', label=f"Optimali kaina: €{optimal_price:.2f}")
plt.title("📊 Kainos įtaka pelnui ir paklausai")
plt.xlabel("Kaina (€)")
plt.ylabel("Pelnas / Paklausa")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Konsolėje ===
print(f"✅ Optimali kaina: €{optimal_price:.2f}")
print(f"📈 Tikėtinas pelnas: €{optimal_profit:.2f}")
print(f"🛒 Tikėtina paklausa: {int(optimal_demand)} vnt")