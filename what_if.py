# what_if_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import copy
from stable_baselines3 import DQN
from env.ecommerce_env_real import AdvancedPricingEnv

# === 1. Setup ===
model_path = "models/dqn/DQN_run_real/dqn_pricing_model"
audit_path = "data/generated/eval_output_dqn_real.csv"
data_path = "data/generated/weekly_env_data_filled.csv"


# === 2. Load environment and model ===
model = DQN.load(model_path)
base_env = AdvancedPricingEnv(data_path=data_path, verbose=False)

# === 3. Load trajectory output ===
df = pd.read_csv(audit_path)
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")
df = df.merge(
    translation[['translated_name', 'main_category']],
    how='left',
    left_on='category',
    right_on='translated_name'
)

# === 4. Simulate all actions at each step ===
def simulate_all_actions(env, current_step):
    results = []
    env.reset()  # ← būtina!
    env.current_step = current_step

    obs = env._get_obs()
    for a in range(env.action_space.n):
        env_copy = copy.deepcopy(env)
        env_copy.current_step -= 1  # kompensuojam
        _, reward, _, _, info = env_copy.step(a)

        reward_components = info.get("reward_components", {})
        profit = reward_components.get("profit", np.nan)

        results.append({
            "action": a,
            "sim_reward": reward,
            "sim_profit": profit,
            "sim_sales": info.get("quantity_sold", np.nan),
            "sim_demand": info.get("total_demand", np.nan),
            "price_elasticity": info.get("price_elasticity", np.nan),
            "stock": info.get("stock", np.nan),
            "category": info.get("product_category_name", "unknown")
        })
    return pd.DataFrame(results)

# === 5. Run what-if analysis ===
policy_vs_best = []
obs = base_env.reset()[0]

for i, row in df.iterrows():
    action_taken = row["action"]
    env_state = copy.deepcopy(base_env)
    sim_df = simulate_all_actions(env_state, current_step=row["step"])

    sim_df["step"] = row["step"]
    sim_df["actual_action"] = action_taken
    sim_df["actual_reward"] = row["reward"]

    best_row = sim_df.loc[sim_df["sim_reward"].idxmax()]
    regret = best_row["sim_reward"] - sim_df.loc[sim_df["action"] == action_taken, "sim_reward"].values[0]

    sim_df["regret"] = regret
    sim_df["best_action"] = best_row["action"]

    policy_vs_best.append(sim_df)

    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = base_env.step(action)
    if done:
        obs = base_env.reset()[0]

# === 6. Save output ===
full_df = pd.concat(policy_vs_best)

# Prijungiam main_category dar kartą, bet jau prie full_df
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")
translation.columns = translation.columns.str.strip()  # remove \n if any
full_df = full_df.merge(
    translation[['translated_name', 'main_category']],
    how='left',
    left_on='category',
    right_on='translated_name'
)


os.makedirs("data/generated", exist_ok=True)
full_df.to_csv("data/generated/what_if_analysis.csv", index=False)
print("✅ Saved what-if action simulation to data/generated/what_if_analysis.csv")

# === 7. Accuracy & Regret Distribution ===
summary = full_df.groupby("step").apply(
    lambda x: x.loc[x["sim_reward"].idxmax(), "action"] == x["actual_action"].iloc[0]
)
accuracy_pct = summary.mean() * 100
print(f"🎯 Agent chose best action in {accuracy_pct:.2f}% of steps")

plt.figure(figsize=(8, 5))
full_df.groupby("step")["regret"].mean().plot.box()
plt.title("Distribution of Regret (Reward Lost Compared to Best Action)")
plt.ylabel("Regret")
plt.tight_layout()
os.makedirs("analysis/LAST_PUSH", exist_ok=True)
plt.savefig("analysis/LAST_PUSH/what_if_regret_distribution.png")
plt.close()

# === 8. Heatmap: actual vs best action ===
conf_matrix = pd.crosstab(full_df["actual_action"], full_df["best_action"], rownames=["Actual"], colnames=["Best"])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("🔍 Confusion Matrix: Actual vs Best Action")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/confusion_action_matrix.png")
plt.close()
print("✅ Saved confusion heatmap to analysis/LAST_PUSH/confusion_action_matrix.png")

# === 9. Regret vs Features ===
plt.figure(figsize=(8, 5))
sns.scatterplot(data=full_df, x="price_elasticity", y="regret")
plt.title("📉 Regret vs Price Elasticity")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/regret_vs_elasticity.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=full_df, x="stock", y="regret")
plt.title("📉 Regret vs Stock")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/regret_vs_stock.png")
plt.close()

# === 9b. Regret by selected categories (closest & furthest from 0) ===
full_df = pd.concat(policy_vs_best)

# Prijungiam main_category dar kartą, bet jau prie full_df
translation = pd.read_csv("data/unique_categories_translated_2.csv", sep=";")
translation.columns = translation.columns.str.strip()  # remove \n if any
full_df = full_df.merge(
    translation[['translated_name', 'main_category']],
    how='left',
    left_on='category',
    right_on='translated_name'
)

def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    return data[(data[column] >= Q1 - 1.5 * IQR) & (data[column] <= Q3 + 1.5 * IQR)]

# Apskaičiuojam vidurkius
category_means = full_df.groupby("translated_name")["regret"].mean().reset_index()
category_means["abs_diff"] = category_means["regret"].abs()

# 6 arčiausiai nulio, 3 toliausiai
closest_cats = category_means.nsmallest(6, "abs_diff")["translated_name"].tolist()
furthest_cats = category_means.nlargest(3, "abs_diff")["translated_name"].tolist()
selected_categories = closest_cats + furthest_cats

n = len(selected_categories)
cols = 3
rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(18, 4 * rows))
axs = axs.flatten()

for i, cat in enumerate(selected_categories):
    data_cat = full_df[full_df["translated_name"] == cat]
    data_cat_clean = remove_outliers_iqr(data_cat, "regret")
    sns.boxplot(data=data_cat_clean, y="regret", ax=axs[i])
    axs[i].set_title(cat)
    axs[i].set_ylabel("Regret")

# Išjungiame tuščius subplot'us
for j in range(i + 1, len(axs)):
    axs[j].axis("off")

plt.suptitle("Regret Distribution by Selected Categories (Cleaned)", fontsize=16)
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/regret_by_selected_categories.png")
plt.close()
print("✅ Saved regret plots for selected categories")

# === 10. Precision per action ===
precision_df = full_df.groupby("actual_action").apply(
    lambda x: (x["actual_action"] == x["best_action"]).mean()
).reset_index(name="precision")

plt.figure(figsize=(8, 5))
sns.barplot(data=precision_df, x="actual_action", y="precision")
plt.title("🎯 Precision per Action (How Often Was It Best)")
plt.ylabel("Precision")
plt.xlabel("Action")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/action_precision.png")
plt.close()
print("✅ Saved precision per action plot to analysis/LAST_PUSH/action_precision.png")