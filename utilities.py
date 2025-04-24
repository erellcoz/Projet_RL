import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


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
