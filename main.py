from env.ecommerce_env import EcommercePricingEnv

env = EcommercePricingEnv()

obs = env.reset()
done = False
total_reward = 0

while not done:
    action = env.action_space.sample()  # random action
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break
    env.render()

print(f"✅ Total simulated revenue: {total_reward:.2f}")