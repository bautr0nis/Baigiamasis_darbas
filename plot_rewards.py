import pandas as pd
import matplotlib.pyplot as plt

# Load reward log
df = pd.read_csv("models/dqn/DQN_run1/rewards.csv")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['reward'], label='Episode Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Over Time (DQN)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/dqn/DQN_run1/reward_plot.png")
plt.show()