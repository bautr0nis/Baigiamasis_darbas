from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env.ecommerce_env_all2 import AdvancedPricingEnv
import os
from itertools import product

# === 1. Grid Parameters ===
param_grid = {
    "exploration_fraction": [0.05, 0.1, 0.15],
    "exploration_final_eps": [0.01],
    "gamma": [0.75, 0.8, 0.85],
    "buffer_size": [50000],
    "learning_rate": [1e-4, 1e-3, 2e-3]
}

param_names = list(param_grid.keys())
grid = list(product(*param_grid.values()))

# === 2. Setup Global Directories ===
BASE_RUN_NAME = "DQN_grid_search"
base_model_dir = f"models/dqn/{BASE_RUN_NAME}"
base_log_dir = "logs/dqn"
tensorboard_base = "./tensorboard"

os.makedirs(base_model_dir, exist_ok=True)
os.makedirs(base_log_dir, exist_ok=True)
os.makedirs(tensorboard_base, exist_ok=True)

# === 3. Run Grid Search ===
for i, values in enumerate(grid):
    run_name = f"{BASE_RUN_NAME}_{i:03d}"
    model_dir = os.path.join(base_model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    # Unpack current hyperparameters
    config = dict(zip(param_names, values))
    print(f"\n🚀 Training model {i+1}/{len(grid)} with config: {config}")

    # === 4. Create Environment ===
    env = DummyVecEnv([
        lambda: Monitor(
            AdvancedPricingEnv(data_path="data/generated/weekly_env_data_filled.csv", verbose=False),
            filename=os.path.join(base_log_dir, f"monitor_{run_name}.csv")
        )
    ])

    # === 5. Initialize Model ===
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=1000,
        batch_size=64,
        gamma=config["gamma"],
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        verbose=0,
        tensorboard_log=tensorboard_base
    )

    # === 6. Train Model ===
    TIMESTEPS = 100_000  # Shorter for grid search
    model.learn(total_timesteps=TIMESTEPS, tb_log_name=run_name)

    # === 7. Save Model ===
    model_path = os.path.join(model_dir, "dqn_pricing_model")
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")

print("\n✅✅ All grid search runs completed.")