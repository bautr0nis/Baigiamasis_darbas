import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import DQN
from env.ecommerce_env_all import AdvancedPricingEnv

# Įkeliame modelį
model = DQN.load("models/compare/real2/DQN_pricing_model.zip")
model.q_net.eval()

# Aplinkos paruošimas
env = AdvancedPricingEnv(data_path="data/generated/weekly_env_data_filled.csv", verbose=False)

# Feature pavadinimai (rankiniu būdu, atitinkantys tavo observation struktūrą)
base_features = [
    "avg_price", "base_price", "base_demand",
    "price_elasticity", "stock", "previous_reward", "demand_level"
]
category_names = env.encoder.get_feature_names_out(["translated_name"])
feature_names = base_features + list(category_names)

# Surenkame stebėjimus
obs_list = []
obs = env.reset()[0]
for _ in range(100):
    obs_list.append(obs)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)
    if done:
        obs = env.reset()[0]

X = np.array(obs_list)

# SHAP predict funkcija
def predict_fn(x):
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        q_vals = model.q_net(x)
        actions = torch.argmax(q_vals, dim=1)
        return q_vals[np.arange(len(q_vals)), actions].numpy()

# SHAP paaiškinimai
explainer = shap.Explainer(predict_fn, X)
shap_values = explainer(X)

# Priskiriam reikšmių vardus
shap_values.feature_names = feature_names

# Vizualizacija
plt.figure(figsize=(12, 10))  # Pakeisk dydį pagal poreikį
shap.plots.bar(shap_values[0], show=False)
plt.tight_layout()
plt.show()

###########################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Nuskaitome modelio evaluacijos rezultatus ===
df = pd.read_csv("data/generated/eval_outputs/real2/eval_output_DQN.csv")

# === 2. Apsiribojame kiekybiniais stulpeliais ir reward ===
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
# Pašaliname nereikalingus arba ID tipo stulpelius jei tokių būtų
excluded_cols = ["step", "action"]  # pridėk daugiau jei reikia
numerical_cols = [col for col in numerical_cols if col not in excluded_cols]

# === 3. Koreliacijų matrica (tik su reward) ===
corr_with_reward = df[numerical_cols].corr()["reward"].drop("reward").sort_values(ascending=False)

# === 4. Heatmap vizualizacija ===
plt.figure(figsize=(10, 6))
sns.heatmap(corr_with_reward.to_frame(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Koreliacija tarp požymių ir reward")
plt.tight_layout()

# === 5. Išsaugome
os.makedirs("analysis/test", exist_ok=True)
plt.savefig("analysis/test/reward_correlation_heatmap.png")
plt.close()

print("✅ Koreliacijų heatmap išsaugotas: analysis/test/reward_correlation_heatmap.png")