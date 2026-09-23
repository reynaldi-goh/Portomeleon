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

    def __init__(self, capacity=50000):
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


class MarginScaler:
    """Turns a raw Q-value margin into a 0-1 confidence by ranking it against
    recent margins: "how sure is this decision compared to the bot's usual ones?"

    This replaces the fixed 0.02 scale, so trade size no longer depends on how
    big the Q-values happen to be (which changed with alpha).
    Keep it with the trained model, and freeze() it after training so the
    frozen model sizes trades against the margins it ended training with.
    """

    def __init__(self, capacity=2000, min_samples=30):
        self.capacity = capacity
        self.min_samples = min_samples
        self.buf = np.zeros(capacity)   # rolling window of recent margins
        self.count = 0                  # how many margins are stored (max = capacity)
        self.next_slot = 0
        self.frozen = False

    def update(self, margin):
        # remember this margin, overwriting the oldest once the window is full
        if self.frozen:
            return
        self.buf[self.next_slot] = margin
        self.next_slot = (self.next_slot + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def confidence(self, margin):
        # share of remembered margins that are <= this one (0 = weakest, 1 = strongest)
        if self.count < self.min_samples:
            return 0.5  # not enough history yet, use a neutral middle size
        return float(np.mean(self.buf[:self.count] <= margin))

    def freeze(self):
        self.frozen = True


# track margins and trade sizes seen during training, for debugging
_margin_log = []
_size_log = []


def margin_to_size(margin, min_size=0.01, max_size=0.90, scale=0.02):
    """Maps a non-negative Q-value margin to a trade-size fraction."""
    confidence = min(margin / scale, 1.0)
    return min_size + confidence * (max_size - min_size)


def select_action_and_size(state, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90,
                            log_margin=False, scaler=None):
    """Epsilon-greedy action choice, plus a confidence-based trade size, for a single stock.

    scaler: a MarginScaler. If given, trade size comes from how this margin ranks
    against recent margins. If None, the old fixed scale of 0.02 is used.
    """
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

    if scaler is not None:
        # rank this margin against recent ones, then remember it (no-op if frozen)
        confidence = scaler.confidence(margin)
        scaler.update(margin)
        size_fraction = min_size + confidence * (max_size - min_size)
    else:
        size_fraction = margin_to_size(margin, min_size=min_size, max_size=max_size, scale=0.02)

    # record the raw margin and the size it produced, for the training printout
    if log_margin:
        _margin_log.append(margin)
        _size_log.append(size_fraction)

    return action, size_fraction


def get_margin_stats():
    # summarize collected margins since the last reset, for debug printing
    if not _margin_log:
        return None
    arr = np.array(_margin_log)
    sizes = np.array(_size_log)
    return {
        "count": len(arr),
        "min": arr.min(),
        "mean": arr.mean(),
        "max": arr.max(),
        "pct_saturated": float(np.mean(arr >= 0.02)),  # raw margins at or above the old fixed cap
        "mean_size": float(sizes.mean()),
        "pct_near_max_size": float(np.mean(sizes >= 0.80)),  # trades sized at 80% or more
    }


def reset_margin_log():
    # clear collected margins and sizes, so stats can be reported per-combo rather than cumulative
    _margin_log.clear()
    _size_log.clear()


def select_portfolio_actions(obs_dict, q_net, epsilon, n_actions, min_size=0.01, max_size=0.90,
                              log_margin=False, scaler=None):
    """Runs select_action_and_size once per ticker, using the same trained network."""
    actions = {}
    for ticker, state in obs_dict.items():
        actions[ticker] = select_action_and_size(
            state, q_net, epsilon, n_actions, min_size=min_size, max_size=max_size,
            log_margin=log_margin, scaler=scaler
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