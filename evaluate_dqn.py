from stable_baselines3 import DQN
from env.ecommerce_env import EcommercePricingEnv

# === RL AGENT ===
env_rl = EcommercePricingEnv()
model = DQN.load("models/dqn/DQN_run1/dqn_pricing_model")

obs = env_rl.reset()[0]
done = False
total_reward_rl = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env_rl.step(action)
    total_reward_rl += reward

print(f"🤖 RL Agent Total Reward: {total_reward_rl:.2f}")

# === BASELINE AGENT (always action=1) ===
env_base = EcommercePricingEnv()
obs = env_base.reset()[0]
done = False
total_reward_base = 0

while not done:
    action = 1  # fixed price (do nothing)
    obs, reward, done, _, _ = env_base.step(action)
    total_reward_base += reward

print(f"📊 Baseline Agent (action=1) Total Reward: {total_reward_base:.2f}")

# === DIFFERENCE ===
diff = total_reward_rl - total_reward_base
percent = (diff / total_reward_base) * 100

print(f"\n📈 Improvement: +{diff:.2f} ({percent:.2f}%) vs Baseline")