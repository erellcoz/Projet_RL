import pickle
import gymnasium as gym
import highway_env

config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "vehicles_count": 6,
    "duration": 11,  # [s]
    "initial_spacing": 0,
    # The reward received when colliding with a vehicle.
    "collision_reward": -10,
    # The reward received when driving on the right-most lanes, linearly mapped to
    "right_lane_reward": 0,
    # zero for other lanes.
    # The reward received when driving at full speed, linearly mapped to zero for
    "high_speed_reward": 0.1,
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
        # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    ],
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 600,  # [px]
    "centering_position": [0.5, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": True,
}

with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)


def make_env():
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_dict)  # Appliquer la configuration
    return env
