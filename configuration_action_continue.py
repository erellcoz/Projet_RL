"""This file runs the REINFORCE algorithm for continuous action spaces on a highway environment."""

import os
import pickle
import gymnasium as gym
import matplotlib.pyplot as plt
from agent_continuous import REINFORCEContinuous
import highway_env
from utilities import run_N_episodes, test_with_reinforce_agent, plot_test_stats

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 50],
            "vy": [-20, 50],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "ContinuousAction",
    },
    "lanes_count": 5,
    "vehicles_count": 5,
    "duration": 30,  # [s]
    "initial_spacing": 0,
    "collision_reward": -10,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0,  # The reward received when driving on the right-most lanes,
    # linearly mapped to zero for other lanes.
    "high_speed_reward": 0,  # The reward received when driving at full speed,
    # linearly mapped to zero for lower speeds according to config["reward_speed_range"].
    "on_road_reward": 4,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly
    # from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)
### Create the environment ###
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.unwrapped.configure(config_dict)
env.reset()
# Actions: discrete action space.
print("Action space:", env.action_space)

# Observations: occupancy grid.
print("Observation space:", env.observation_space)


### Create the agent ###
LEARNING_RATE = 3e-4
GAMMA = 0.99

N_EPISODES = 500
EPISODE_BATCH_SIZE = 15

agent = REINFORCEContinuous(
    env.action_space,
    env.observation_space,
    episode_batch_size=EPISODE_BATCH_SIZE,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
)


# Create folder for saving results
FOLDER = "results/continuous/"
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)


### Run the agent ###
env, agent, reward_per_episode, _ = run_N_episodes(
    env, agent, N_episodes=N_EPISODES, display=False
)

# Save agent
with open(os.path.join(FOLDER, "agent.pkl"), "wb") as f:
    pickle.dump(agent, f)


# Plot the training results
plt.figure(figsize=(20, 10))
plt.subplot(221)
plt.plot(reward_per_episode)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.subplot(222)
plt.plot(agent.losses.values())
plt.xlabel("Episode batch number")
plt.ylabel("Loss")
plt.subplot(223)
plt.plot(agent.means)
plt.xlabel("Episode")
plt.ylabel("Mean parameter of policy")
plt.subplot(224)
plt.plot(agent.stds)
plt.xlabel("Episode")
plt.ylabel("Std parameter of policy")
plt.savefig(os.path.join(FOLDER, "training_results.png"))
plt.show()

### Test the agent ###
env = test_with_reinforce_agent(agent, env, n_episodes=10, display=True)

### Plot the testing results ###
plot_test_stats(env, agent, plot=True)
