import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from torch import nn
from copy import deepcopy

def run_N_episodes(env, agent, N_episodes=1, display=False):
    reward_per_episode = []
    step_per_episode = []

    for i in range(N_episodes):
        tot_reward = 0
        N_steps = 0
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, terminated, next_state)
            done = terminated or truncated
            state = next_state
            tot_reward += reward
            N_steps += 1

            # Display the environment
            if display:
                env.render()

        # Pause at the end of the episode to visualize the ending state
        if display:
            plt.pause(0.5)

        if i % 10 == 0:
            print(f"Episode {i} finished")

        # Store the total reward for this episode
        reward_per_episode.append(tot_reward)
        step_per_episode.append(N_steps)

    return env, agent, reward_per_episode, step_per_episode


# Test the greedy policy for the q-function learned by the agent
def test(agent, env, n_episodes=100, display=False):
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.eps_greedy(obs, eps=0)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if display:
                env.render()
            done = terminated or truncated
            obs = next_obs
        if display:
            plt.pause(0.5)

    return env


def test_sb3(model, env, n_episodes=100, display=False):
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if display:
                env.render()
            done = terminated or truncated
        if display:
            plt.pause(0.5)
    return env


def plot_test_stats(env, agent, plot=False):
    returns = np.array(env.return_queue).flatten()
    N_ep = len(returns)
    print('Mean return over n={} episodes : {} +/- {}'.format(N_ep,
          np.mean(returns), np.std(returns)/np.sqrt(N_ep)))
    ep_lengths = np.array(env.length_queue).flatten()
    print('Mean episode length over n={} episodes : {}'.format(
        N_ep, np.mean(ep_lengths), np.std(ep_lengths)/np.sqrt(N_ep)))

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
        axs[0].set_title("Episode rewards")
        axs[0].set_xlabel("Episode rewards")
        axs[0].set_ylabel("Frequency")
        axs[0].axvline(x=np.mean(returns), linestyle='--', alpha=0.5)
        axs[0].legend(['Mean reward'])
        axs[0].hist(returns)

        axs[1].set_title("Episode lengths")
        axs[1].set_xlabel("Episode lengths")
        axs[1].set_ylabel("Frequency")
        axs[1].hist(ep_lengths)
        axs[1].axvline(x=np.mean(ep_lengths))
        axs[1].legend(['Mean episode length'])

        plt.tight_layout()
        plt.show()


def rewardseq_to_returns(reward_list: list[float], gamma: float) -> list[float]:
    """
    Turns a list of rewards into the list of returns
    """
    G = 0
    returns_list = []
    for r in reward_list[::-1]:
        G = r + gamma * G
        returns_list.append(G)
    return returns_list[::-1]


def to_arr(value_dict):
    res = np.zeros(7)
    for state in value_dict:
        res[state] = value_dict[state]

    return res


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)

# create instance of replay buffer
#replay_buffer = ReplayBuffer(1000)


class Net(nn.Module):
    """
    Basic neural net.
    """
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)
    

def eval_agent(agent, env, n_sim=5, display=False):
        """        
        Monte Carlo evaluation of DQN agent.

        Repeat n_sim times:
            * Run the DQN policy until the environment reaches a terminal state (= one episode)
            * Compute the sum of rewards in this episode
            * Store the sum of rewards in the episode_rewards array.
        """
        env_copy = deepcopy(env)
        episode_rewards = np.zeros(n_sim)
        episode_steps = np.zeros(n_sim)
        episode_collision_rewards = np.zeros(n_sim)
        episode_right_lane_rewards = np.zeros(n_sim)
        episode_high_speed_rewards = np.zeros(n_sim)
        for i in range(n_sim):
            state, _ = env_copy.reset()
            reward_sum = 0
            N_steps = 0
            collision_reward = 0
            right_lane_reward = 0
            high_speed_reward = 0
            done = False
            while not done: 
                action = agent.get_action(state, 0)
                state, reward, terminated, truncated, info = env_copy.step(action)
               
                rewards = info["rewards"]
                collision_reward += rewards["collision_reward"]
                right_lane_reward += rewards["right_lane_reward"]
                high_speed_reward += rewards["high_speed_reward"]
                
                reward_sum += reward
                N_steps += 1
                done = terminated or truncated
                
                # Display the environment
                if display:
                    env_copy.render()
                    
            episode_rewards[i] = reward_sum
            episode_steps[i] = N_steps
            episode_collision_rewards[i] = collision_reward
            episode_right_lane_rewards[i] = right_lane_reward
            episode_high_speed_rewards[i] = high_speed_reward
            
            # Pause at the end of the episode to visualize the ending state
            if display:
                plt.pause(1)  # Pause for 1 seconds
                
        return episode_rewards , episode_steps, episode_collision_rewards, episode_right_lane_rewards, episode_high_speed_rewards
    
    
def train_dqn(env, agent, N_episodes, eval_every=10, reward_threshold=300):
    total_time = 0
    state, _ = env.reset()
    losses = []
    reward_per_episode = []
    step_per_episode = []
    eps_decrease = []
    
    for ep in range(N_episodes):
        done = False
        state, _ = env.reset()
        tot_reward = 0
        N_step = 0
        episode_losses = []
        while not done: 
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            loss_val = agent.update(state, action, reward, terminated, next_state)

            eps_decrease.append(agent.epsilon)
            state = next_state
            episode_losses.append(loss_val)
            N_step += 1
            tot_reward += reward

            done = terminated or truncated
            total_time += 1

        if ((ep+1)% eval_every == 0):
            rewards, episode_steps, _, _, _ = eval_agent(agent, env)
            print("episode =", ep+1, ", reward = ", np.mean(rewards), ", episode length = ", np.mean(episode_steps))
            if np.mean(rewards) >= reward_threshold:
                break
        
        losses.append(episode_losses)
        reward_per_episode.append(tot_reward)
        step_per_episode.append(N_step)
                
    return losses, reward_per_episode, step_per_episode, eps_decrease