import pickle
import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from agent_DQN import DQN
from utilities import eval_agent, train_dqn
import numpy as np 
import json
import os

### Configuration ###
config_dict = {
    "observation": {
        "type": "OccupancyGrid",  # For each observed feature, the grid is a 3D array of size 
                                  # (N_features, N_side_cells, N_side_cells) indicating the value of  
                                  # the feature in each cell of the grid. The ego-vehicle is  
                                  # implicitly at the center of the observation grid.
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],  # 7 features
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],   # physical size of the grid in meters (x and y dimensions).
        "grid_step": [5, 5],    # The resolution of the grid in meters per cell. 
                                # This results in an 8 x 8 grid because: 
                                # Number of side cells = grid_size_range / grid step 
                                #                      = 40 / 5 = 8 
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction", },  # 5 actions: { 0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT',
                                          #              3: 'FASTER', 4: 'SLOWER'
    "lanes_count": 4,
    "vehicles_count": 15,
    "duration": 60,  # [s]
    "initial_spacing": 0,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],   # [x, y]: The position of the center of the screen in the rendering view.
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

# Create the agent
action_space = env.action_space
observation_space = env.observation_space

gamma = 0.8
batch_size = 32
buffer_capacity = 15_000
update_target_every = 50

epsilon_start = 0.9
decrease_epsilon_factor = 1000
epsilon_min = 0.05

learning_rate = 5e-4

arguments = (env,
            action_space,
            observation_space,
            gamma,
            batch_size,
            buffer_capacity,
            update_target_every, 
            epsilon_start, 
            decrease_epsilon_factor, 
            epsilon_min,
            learning_rate,
        )

N_episodes = 3000

agent = DQN(*arguments)

# Create folder for saving results
folder = "results/initiale_config/initialeconfig5"
if not os.path.exists(folder):
    os.makedirs(folder)
    
# Save parameters in JSON
params = {
    "N_episodes": N_episodes,
    "gamma": gamma,
    "batch_size": batch_size,
    "buffer_capacity": buffer_capacity,
    "update_target_every": update_target_every,
    "epsilon_start": epsilon_start,
    "decrease_epsilon_factor": decrease_epsilon_factor,
    "epsilon_min": epsilon_min,
    "learning_rate": learning_rate,
    "high_speed_reward": config_dict["high_speed_reward"],
    "collision_reward": config_dict["collision_reward"],
    "right_lane_reward": config_dict["right_lane_reward"],
}

with open(os.path.join(folder, "params.json"), "w") as f:
    json.dump(params, f, indent=4)
      
# Run the training loop
losses, reward_per_episode, step_per_episode, eps_decrease = train_dqn(env, agent, N_episodes)

# Save agent
with open(os.path.join(folder,"agent.pkl"), "wb") as f:
    pickle.dump(agent, f)

# Save training statistics
with open(os.path.join(folder,"training_stats.pkl"), "wb") as f:
    pickle.dump((losses, reward_per_episode, step_per_episode, eps_decrease), f)
    
# Plot and save training statistics
losses_mean = [np.mean(episode_losses) for episode_losses in losses]
plt.plot(losses_mean)
plt.xlabel("Episode")
plt.ylabel("Mean Loss")
plt.savefig(os.path.join(folder, "losses.png"))
plt.show()

reward_mean_per_10_episodes = [np.mean(reward_per_episode[i:i+10]) for i in range(0, len(reward_per_episode), 10)]
plt.plot(reward_mean_per_10_episodes)
plt.xlabel("Episode")
plt.ylabel("Mean Reward per 10 episodes")
#plt.xticks(ticks=range(0, len(reward_mean_per_10_episodes)), labels=range(10, len(reward_mean_per_10_episodes) * 10 + 1, 10))
plt.savefig(os.path.join(folder, "rewards.png"))
plt.show()

length_mean_per_10_episodes = [np.mean(step_per_episode[i:i+10]) for i in range(0, len(step_per_episode), 10)]
plt.plot(length_mean_per_10_episodes)
plt.xlabel("Episode")
plt.ylabel("Mean Episode Length per 10 episodes")
#plt.xticks(ticks=range(0, len(reward_mean_per_10_episodes)), labels=range(10, len(reward_mean_per_10_episodes) * 10 + 1, 10))
plt.savefig(os.path.join(folder, "lengths.png"))
plt.show()

# Plot epsilon decay
plt.plot(eps_decrease)
plt.xlabel("Step")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")
plt.savefig(os.path.join(folder, "epsilon_decay.png"))
plt.show()

# Evaluate the final policy
rewards, episode_steps, _, _, _ = eval_agent(agent, env, 10, display=True)
print("")
print("mean reward after training = ", np.mean(rewards))
print("mean episode steps after training = ", np.mean(episode_steps))