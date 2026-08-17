"""Agent: Q-network, replay buffer, action selection, and training step — Iteration 3."""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Maps a state to a Q-value for each action."""

    # create the neural network architecture, take observation, such as close, MA10,
    # output q values of actions
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    # calculate Q values from a state
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores past experiences and samples random batches for training."""

    # create memory for storing experiences, default size = 10000 elements
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # auto-drops oldest when full

    # save one experience to replay buffer
    def push(self, state, action, reward, next_state, done):
        # save one experience tuple
        self.buffer.append((state, action, reward, next_state, done))

    # get a random batch of experiences
    def sample(self, batch_size):
        # random batch, breaks day-to-day correlation
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    # return the number of stored experiences
    def __len__(self):
        return len(self.buffer)


# convert Q value to confidence in trade size
# never trade below 1% or over 90% cash
def margin_to_size(margin, min_size=0.01, max_size=0.90, scale=0.02):
    """Maps a non-negative Q-value margin to a trade-size fraction."""
    confidence = min(margin / scale, 1.0)
    return min_size + confidence * (max_size - min_size)


# choose an action and trade size for ONE stock, using epsilon greedy strategy
def select_action_and_size(state, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90):
    """Epsilon-greedy action choice, plus a confidence-based trade size, for a single stock."""
    # when the epsilon is still big
    if random.random() < epsilon:
        action = random.randint(0, n_actions - 1)  # pick a random action
        size_fraction = random.uniform(min_size, max_size)  # pick a random size
        return action, size_fraction
    # disable gradient tracking during inference
    with torch.no_grad():
        # convert state in tensor format
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # predict q values for all actions
        q_values = q_net(state_tensor).squeeze(0)
        # choose action with the highest q value
        action = torch.argmax(q_values).item()
    # hold actions
    if action == 0:
        return action, 0.0
    # find second best action
    other_actions = [i for i in range(n_actions) if i != action]
    # get runner up q value, if it is buy, then the runner up is sell and vice versa
    runner_up_q = max(q_values[i].item() for i in other_actions)
    # take difference of q value
    margin = q_values[action].item() - runner_up_q
    # convert difference to confidence level
    # the bigger the difference, the more confidence bot has over the best action
    size_fraction = margin_to_size(margin, min_size=min_size, max_size=max_size, scale=0.02)
    return action, size_fraction


# choose an action and trade size for EVERY stock in the portfolio, for one day
def select_portfolio_actions(obs_dict, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90):
    """
    Runs select_action_and_size once per ticker, using the same trained network.

    obs_dict: {ticker: state_vector}, as returned by TradingEnv._get_obs()

    Returns: {ticker: (action, size_fraction)}, ready to pass into TradingEnv.step()
    """
    actions = {}
    for ticker, state in obs_dict.items():
        actions[ticker] = select_action_and_size(
            state, q_net, epsilon, n_actions, min_size=min_size, max_size=max_size
        )
    return actions


# train the q network using a batch of past experiences
def train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=32, gamma=0.99):
    """One gradient update from a random batch of past experiences."""
    if len(buffer) < batch_size:
        return None  # not enough memories yet

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # Q value the network predicted for the action actually taken
    q_values = q_net(states)
    predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # best possible next-state value, from the stable target network
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]

    # Bellman target: reward now plus discounted future value
    target_q = rewards + gamma * max_next_q * (1 - dones)

    # calculate loss and update network weights
    loss = loss_fn(predicted_q, target_q)
    # clear old gradients
    optimizer.zero_grad()
    # compute gradients
    loss.backward()
    # update network weights
    optimizer.step()

    return loss.item()


# copy weights from q network to the target network
def update_target_network(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())


# create all components needed for the dqn agent
def build_agent(obs_size, n_actions, lr=1e-3):
    """Creates q_net, target_net, optimizer, and loss function together."""
    # create current network
    q_net = QNetwork(obs_size, n_actions)
    # create frozen, target network
    target_net = QNetwork(obs_size, n_actions)
    # start both networks with identical weights
    target_net.load_state_dict(q_net.state_dict())
    # only ever updated by copying, never by backprop
    target_net.eval()
    # adam optimiser for gradient updates
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    # huber loss
    loss_fn = nn.SmoothL1Loss()

    return q_net, target_net, optimizer, loss_fn