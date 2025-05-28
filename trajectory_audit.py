# trajectory_audit_plot.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1. Load trajectory audit data ===
audit_path = "data/generated/eval_output_dqn_real.csv"
df = pd.read_csv(audit_path)

# === 2. Normalize reward components ===
reward_components = [col for col in df.columns if col.startswith("reward_components.")]

if not reward_components:
    print("⚠️ No reward components found. Make sure you used env.get_trajectory_log() with reward_components in info.")
    exit()

# Normalize JSON-style columns if needed
if isinstance(df[reward_components[0]][0], str):
    df_components = pd.json_normalize(df["reward_components"].apply(eval))
    df = pd.concat([df.drop(columns=["reward_components"]), df_components], axis=1)
    reward_components = list(df_components.columns)

# === 3. Clean column names ===
clean_names = {col: col.replace("reward_components.", "") for col in reward_components}
df.rename(columns=clean_names, inplace=True)
reward_components = [clean_names[col] for col in reward_components]

# === 4. Create summary table ===
avg_contributions = df[reward_components].mean().sort_values(ascending=False)
sum_contributions = df[reward_components].sum().sort_values(ascending=False)

summary_df = pd.DataFrame({
    "mean_contribution": avg_contributions,
    "total_contribution": sum_contributions
})

os.makedirs("analysis/LAST_PUSH", exist_ok=True)
summary_df.to_csv("analysis/LAST_PUSH/reward_component_summary.csv")
print("✅ Saved reward component summary table to analysis/LAST_PUSH/reward_component_summary.csv")

# === 5. Total contribution bar chart ===
plt.figure(figsize=(12, 6))
sum_contributions.plot(kind="barh")
plt.title("🔍 Total Contribution of Reward Components")
plt.xlabel("Total Contribution to Reward")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/reward_component_totals.png")
plt.close()
print("✅ Saved total contribution bar chart to analysis/LAST_PUSH/reward_component_totals.png")

# === 6. Correlation with final reward ===
plt.figure(figsize=(10, 6))
corr = df[reward_components + ["reward"]].corr()["reward"].drop("reward").sort_values()
sns.barplot(x=corr.values, y=corr.index, palette="coolwarm")
plt.title("Correlation of Reward Components with Final Reward")
plt.xlabel("Correlation with Total Reward")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/reward_component_vs_total_correlation.png")
plt.close()
print("✅ Saved correlation plot to analysis/LAST_PUSH/reward_component_vs_total_correlation.png")

########################################################

# === 7. SHAP-like Global Feature Importance Plot ===
plt.figure(figsize=(10, 6))

# Naudosime absoliučius mean contributions kaip svarbos metriką
shap_like_importance = df[reward_components].abs().mean().sort_values(ascending=True)

sns.barplot(x=shap_like_importance.values, y=shap_like_importance.index, palette="viridis")

plt.title("🌟 SHAP-like Global Feature Importance (Mean Abs Contribution)")
plt.xlabel("Mean Absolute Contribution to Reward")
plt.ylabel("Reward Component")
plt.tight_layout()
plt.savefig("analysis/LAST_PUSH/shap_like_global_importance.png")
plt.close()

print("✅ Saved SHAP-like global feature importance plot to analysis/LAST_PUSH/shap_like_global_importance.png")