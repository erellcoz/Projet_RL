"""This file contains the implementation of the REINFORCE algorithm for continuous action spaces."""

# Imports
import torch
from torch import nn
from torch import optim
import numpy as np


class NetContinousActions(nn.Module):
    """
    Basic neural net for continuous actions.
    """

    def __init__(self, obs_dim, hidden_size, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim),
        )
        self.log_std = nn.Parameter(
            torch.zeros(act_dim)
        )  # learnable log std for Gaussian policy

    def forward(self, obs):
        """
        Forward pass through the network.
        """
        mean = self.net(obs)
        # We clamp the log_std to avoid that the std is too small or too big.
        # This is important for the stability of the training.
        log_std = torch.clamp(self.log_std, min=-5, max=2)
        std = log_std.exp()
        return mean, std


class REINFORCEContinuous:
    """
    REINFORCE agent for continuous action spaces.
    """

    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate

        # Reset
        hidden_size = 256

        obs_size = np.prod(self.observation_space.shape)
        actions_dim = self.action_space.shape[0]

        self.policy_net = NetContinousActions(obs_size, hidden_size, actions_dim)

        self.scores = []
        self.current_episode = []
        self.losses = {}
        self.stds = []
        self.means = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0

    def update(self, state, action, reward, done, next_state):
        """
        Update the agent with the current state, action, reward, and done flag.
        This function is called at each step of the environment.
        """
        self.current_episode.append(
            (
                torch.tensor(state).unsqueeze(0),
                torch.tensor(action, dtype=torch.float32).unsqueeze(0),
                torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
            )
        )

        if done:
            self.n_eps += 1

            states, actions, rewards = tuple(
                [torch.cat(data) for data in zip(*self.current_episode)]
            )

            states = states.view(states.shape[0], -1)

            current_episode_returns = self._returns(rewards, self.gamma)
            current_episode_returns = (
                current_episode_returns - current_episode_returns.mean()
            )

            means, stds = self.policy_net.forward(states)
            dist = torch.distributions.Normal(means, stds)
            log_probs = dist.log_prob(actions).sum(dim=-1)

            score = log_probs * current_episode_returns

            self.scores.append(score.sum().unsqueeze(0))
            self.current_episode = []
            self.stds.append(stds.mean().item())
            self.means.append(means.mean().item())

            if (self.n_eps % self.episode_batch_size) == 0:
                self.optimizer.zero_grad()
                full_neg_score = -torch.cat(self.scores).sum() / self.episode_batch_size
                full_neg_score.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                print(f"Loss at episode {self.n_eps}: {full_neg_score.item():.4f}")
                self.losses[self.n_eps] = full_neg_score.item()

                self.scores = []

    def _returns(self, rewards, gamma):
        """
        Compute the returns for a given episode.
        """
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return torch.tensor(returns, dtype=torch.float32)

    def get_action(self, state):
        """
        Get the action for a given state.
        """
        state_tensor = (
            torch.as_tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        )
        reduction_factor_max_acceleration = 1000
        reduction_factor_min_acceleration = 1000
        reduction_factor_steering_angle = 1000

        with torch.no_grad():
            mean, std = self.policy_net(state_tensor)
            action = torch.distributions.Normal(mean, std).sample()

            max_acceleration = (
                self.action_space.high[0] / reduction_factor_max_acceleration
            )
            min_acceleration = (
                self.action_space.low[0] / reduction_factor_min_acceleration
            )

            min_steering = self.action_space.low[1] / reduction_factor_steering_angle
            max_steering = self.action_space.high[1] / reduction_factor_steering_angle

            low = torch.tensor([min_acceleration, min_steering], dtype=torch.float32)
            high = torch.tensor([max_acceleration, max_steering], dtype=torch.float32)

            return action.clamp(low, high).numpy()[0]

    def train_reset(self):
        """
        Reset the agent for training.
        This function is called at the beginning of each training session.
        """
        self.scores = []
        self.current_episode = []

    def reset(self):
        """
        Reset the agent for testing.
        """
        hidden_size = 126

        obs_size = np.prod(self.observation_space.shape)
        actions_dim = self.action_space.shape[0]

        self.policy_net = NetContinousActions(obs_size, hidden_size, actions_dim)

        self.scores = []
        self.current_episode = []

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=self.learning_rate
        )

        self.n_eps = 0
