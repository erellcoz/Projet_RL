import pickle
from utilities import eval_agent
import gymnasium as gym
import highway_env
import numpy as np
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
    "collision_reward": -90,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 1,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.5,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "normalize_reward": True,
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

### Create the environment ###
env = gym.make("highway-fast-v0", render_mode="rgb_array")
env.unwrapped.configure(config_dict)
env.reset()

# Load agent
folder = "results/initiale_config/initialeconfig1"
path = os.path.join(folder, "agent.pkl")
with open(path, "rb") as f:
    agent = pickle.load(f)
    

# Evaluate the final policy
rewards, episode_steps, episode_collision_rewards, episode_right_lane_rewards, episode_high_speed_rewards = eval_agent(agent, env, 20, display=True)
print("")
print(rewards)
print(episode_collision_rewards)
print(episode_right_lane_rewards)
print(episode_high_speed_rewards)
print("mean reward after training = ", np.mean(rewards))
print("mean episode steps after training = ", np.mean(episode_steps))
print("mean episode right lane reward = ", np.mean(episode_right_lane_rewards))
print("mean episode collision reward = ", np.mean(episode_collision_rewards))
print("mean episode high speed reward = ", np.mean(episode_high_speed_rewards))
