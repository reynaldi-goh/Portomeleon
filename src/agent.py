"""Agent: Q-network, replay buffer, action selection, and training step — Iteration 3."""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    # fix every source of randomness so a run can be exactly reproduced
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QNetwork(nn.Module):
    """Maps a state to a Q-value for each action."""

    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Stores past experiences and samples random batches for training."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# track margins seen during training, for debugging the scale parameter below
_margin_log = []


def margin_to_size(margin, min_size=0.01, max_size=0.90, scale=0.02):
    """Maps a non-negative Q-value margin to a trade-size fraction."""
    confidence = min(margin / scale, 1.0)
    return min_size + confidence * (max_size - min_size)


def select_action_and_size(state, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90,
                            log_margin=False):
    """Epsilon-greedy action choice, plus a confidence-based trade size, for a single stock."""
    if random.random() < epsilon:
        action = random.randint(0, n_actions - 1)
        size_fraction = random.uniform(min_size, max_size)
        return action, size_fraction

    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(state_tensor).squeeze(0)
        action = torch.argmax(q_values).item()

    if action == 0:
        return action, 0.0

    other_actions = [i for i in range(n_actions) if i != action]
    runner_up_q = max(q_values[i].item() for i in other_actions)
    margin = q_values[action].item() - runner_up_q

    # record the raw margin so we can check whether scale=0.02 is saturating confidence
    if log_margin:
        _margin_log.append(margin)

    size_fraction = margin_to_size(margin, min_size=min_size, max_size=max_size, scale=0.02)
    return action, size_fraction


def get_margin_stats():
    # summarize collected margins since the last reset, for debug printing
    if not _margin_log:
        return None
    arr = np.array(_margin_log)
    return {
        "count": len(arr),
        "min": arr.min(),
        "mean": arr.mean(),
        "max": arr.max(),
        "pct_saturated": float(np.mean(arr >= 0.02)),  # fraction hitting max confidence
    }


def reset_margin_log():
    # clear collected margins, so stats can be reported per-combo rather than cumulative
    _margin_log.clear()


def select_portfolio_actions(obs_dict, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90,
                              log_margin=False):
    """Runs select_action_and_size once per ticker, using the same trained network."""
    actions = {}
    for ticker, state in obs_dict.items():
        actions[ticker] = select_action_and_size(
            state, q_net, epsilon, n_actions, min_size=min_size, max_size=max_size,
            log_margin=log_margin
        )
    return actions


def train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=32, gamma=0.99):
    """One gradient update from a random batch of past experiences."""
    if len(buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    q_values = q_net(states)
    predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(1)[0]

    target_q = rewards + gamma * max_next_q * (1 - dones)

    loss = loss_fn(predicted_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def update_target_network(q_net, target_net):
    target_net.load_state_dict(q_net.state_dict())


def build_agent(obs_size, n_actions, lr=1e-3, seed=None):
    """Creates q_net, target_net, optimizer, and loss function together.

    seed: if given, fixes all randomness BEFORE the network is initialized,
    so weight initialization itself becomes reproducible too.
    """
    if seed is not None:
        set_seed(seed)

    q_net = QNetwork(obs_size, n_actions)
    target_net = QNetwork(obs_size, n_actions)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    return q_net, target_net, optimizer, loss_fn