import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from env.ecommerce_env import EcommercePricingEnv

# 1. Įkeliam apmokytą modelį
model_path = "models/dqn/DQN_run1/dqn_pricing_model"
model = DQN.load(model_path)

# 2. Paruošiam aplinką
env = EcommercePricingEnv()
observations = []

# 3. Kaupiam stebėjimus
obs, _ = env.reset()
done = False
while not done and len(observations) < 500:
    observations.append(obs)
    action, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env.step(action)

observations = np.array(observations)

# 4. Pasirenkam veiksmą, kurį aiškinsime
target_action_index = 1  # Pvz.: 0 = sumažinti, 1 = palikti, 2 = padidinti

# 5. Sukuriam funkciją, kuri gražina Q reikšmę tik vienam veiksmui
def single_action_q_value(X):
    X_tensor = model.policy.obs_to_tensor(X)[0]
    q_values = model.q_net(X_tensor).detach().cpu().numpy()
    return q_values[:, target_action_index]

# 6. SHAP paaiškinimai
explainer = shap.Explainer(single_action_q_value, observations)
shap_values = explainer(observations)

# 7. Grafikas
shap.summary_plot(shap_values.values, features=observations, feature_names=env.features.columns)
plt.savefig(f"models/dqn/DQN_run1/shap_action_{target_action_index}.png")
plt.show()