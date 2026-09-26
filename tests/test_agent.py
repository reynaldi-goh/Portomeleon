"""Unit tests for the DQN agent: Iteration 3."""
import numpy as np
import torch
from src.agent import (
    QNetwork,
    ReplayBuffer,
    MarginScaler,
    margin_to_size,
    select_action_and_size,
    select_portfolio_actions,
    build_agent,
    train_step,
    update_target_network,
)


# ---------- QNetwork ----------

def test_qnetwork_output_shape():
    net = QNetwork(obs_size=7, n_actions=3)
    state = torch.rand(1, 7)
    q_values = net(state)
    assert q_values.shape == (1, 3)


def test_qnetwork_output_is_finite():
    net = QNetwork(obs_size=7, n_actions=3)
    state = torch.rand(5, 7)  # batch of 5
    q_values = net(state)
    assert torch.isfinite(q_values).all()


# ---------- ReplayBuffer ----------

def test_replay_buffer_push_and_len():
    buffer = ReplayBuffer(capacity=10)
    state = np.zeros(7)
    buffer.push(state, 1, 0.05, state, False)
    assert len(buffer) == 1


def test_replay_buffer_respects_capacity():
    buffer = ReplayBuffer(capacity=5)
    state = np.zeros(7)
    for i in range(10):
        buffer.push(state, 1, 0.01, state, False)
    assert len(buffer) == 5  # oldest entries dropped


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=50)
    state = np.random.rand(7)
    for _ in range(20):
        buffer.push(state, 1, 0.02, state, False)
    states, actions, rewards, next_states, dones = buffer.sample(8)
    assert states.shape == (8, 7)
    assert actions.shape == (8,)
    assert rewards.shape == (8,)
    assert next_states.shape == (8, 7)
    assert dones.shape == (8,)


# ---------- MarginScaler ----------

def test_marginscaler_returns_neutral_confidence_before_min_samples():
    # fewer than min_samples margins seen so far should fall back to 0.5
    scaler = MarginScaler(capacity=100, min_samples=10)
    for _ in range(5):
        scaler.update(0.03)
    assert scaler.confidence(0.03) == 0.5


def test_marginscaler_confidence_ranks_against_history():
    # once enough history exists, confidence should reflect where a margin ranks
    scaler = MarginScaler(capacity=100, min_samples=5)
    for margin in [0.01, 0.02, 0.03, 0.04, 0.05]:
        scaler.update(margin)

    low_confidence = scaler.confidence(0.0)   # below every stored margin
    high_confidence = scaler.confidence(1.0)  # above every stored margin
    assert low_confidence < high_confidence
    assert high_confidence == 1.0


def test_marginscaler_freeze_stops_updates():
    # once frozen, new margins should not change what confidence() reports
    scaler = MarginScaler(capacity=100, min_samples=5)
    for margin in [0.01, 0.02, 0.03, 0.04, 0.05]:
        scaler.update(margin)
    scaler.freeze()

    before = scaler.confidence(0.03)
    scaler.update(100.0)  # should be ignored now that the scaler is frozen
    after = scaler.confidence(0.03)
    assert before == after


def test_marginscaler_count_caps_at_capacity():
    # pushing more margins than capacity should not grow count past capacity
    scaler = MarginScaler(capacity=10, min_samples=1)
    for i in range(50):
        scaler.update(float(i))
    assert scaler.count == 10


# ---------- margin_to_size ----------

def test_margin_to_size_zero_margin_gives_min_size():
    size = margin_to_size(margin=0.0, min_size=0.01, max_size=0.90, scale=0.02)
    assert np.isclose(size, 0.01)


def test_margin_to_size_large_margin_caps_at_max_size():
    size = margin_to_size(margin=10.0, min_size=0.01, max_size=0.90, scale=0.02)
    assert np.isclose(size, 0.90)


def test_margin_to_size_stays_within_bounds():
    for margin in [0.0, 0.005, 0.01, 0.02, 0.05, 1.0]:
        size = margin_to_size(margin, min_size=0.01, max_size=0.90, scale=0.02)
        assert 0.01 <= size <= 0.90


# ---------- select_action_and_size ----------

def test_select_action_and_size_greedy_hold_returns_zero_size():
    # force a network that always prefers hold (action 0)
    net = QNetwork(obs_size=7, n_actions=3)
    with torch.no_grad():
        # bias the output layer so hold's Q-value dominates
        net.net[-1].bias[:] = torch.tensor([10.0, -10.0, -10.0])
        net.net[-1].weight[:] = 0.0

    state = np.random.rand(7)
    action, size_fraction = select_action_and_size(state, net, epsilon=0.0, n_actions=3)
    assert action == 0
    assert size_fraction == 0.0


def test_select_action_and_size_greedy_buy_returns_positive_size():
    net = QNetwork(obs_size=7, n_actions=3)
    with torch.no_grad():
        # bias the output layer so buy's Q-value dominates
        net.net[-1].bias[:] = torch.tensor([-10.0, 10.0, -10.0])
        net.net[-1].weight[:] = 0.0

    state = np.random.rand(7)
    action, size_fraction = select_action_and_size(state, net, epsilon=0.0, n_actions=3)
    assert action == 1
    assert 0.01 <= size_fraction <= 0.90


def test_select_action_and_size_full_exploration_is_random():
    net = QNetwork(obs_size=7, n_actions=3)
    state = np.random.rand(7)
    # epsilon=1.0 always explores, action should be a valid index
    action, size_fraction = select_action_and_size(state, net, epsilon=1.0, n_actions=3)
    assert action in (0, 1, 2)
    assert 0.0 <= size_fraction <= 0.90


def test_select_action_and_size_uses_scaler_when_given():
    # with a scaler passed in, size should come from ranked confidence, not the fixed scale
    net = QNetwork(obs_size=7, n_actions=3)
    with torch.no_grad():
        net.net[-1].bias[:] = torch.tensor([-10.0, 10.0, -10.0])
        net.net[-1].weight[:] = 0.0

    scaler = MarginScaler(capacity=100, min_samples=1000)  # stays in the neutral 0.5 regime
    state = np.random.rand(7)
    action, size_fraction = select_action_and_size(state, net, epsilon=0.0, n_actions=3, scaler=scaler)

    assert action == 1
    assert np.isclose(size_fraction, 0.01 + 0.5 * (0.90 - 0.01))


# ---------- select_portfolio_actions ----------

def test_select_portfolio_actions_covers_every_ticker():
    net = QNetwork(obs_size=7, n_actions=3)
    obs_dict = {
        "AAPL": np.random.rand(7),
        "NVDA": np.random.rand(7),
        "GOOG": np.random.rand(7),
    }
    actions = select_portfolio_actions(obs_dict, net, epsilon=0.0, n_actions=3)
    assert set(actions.keys()) == {"AAPL", "NVDA", "GOOG"}


def test_select_portfolio_actions_returns_valid_action_and_size():
    net = QNetwork(obs_size=7, n_actions=3)
    obs_dict = {"AAPL": np.random.rand(7), "NVDA": np.random.rand(7)}
    actions = select_portfolio_actions(obs_dict, net, epsilon=0.0, n_actions=3)
    for ticker, (action, size_fraction) in actions.items():
        assert action in (0, 1, 2)
        assert 0.0 <= size_fraction <= 0.90


# ---------- train_step ----------

def test_train_step_returns_none_when_buffer_too_small():
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size=7, n_actions=3)
    buffer = ReplayBuffer(capacity=50)
    buffer.push(np.random.rand(7), 1, 0.02, np.random.rand(7), False)  # only 1 entry
    loss = train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=32)
    assert loss is None


def test_train_step_returns_finite_loss():
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size=7, n_actions=3)
    buffer = ReplayBuffer(capacity=50)
    for _ in range(40):
        buffer.push(np.random.rand(7), np.random.randint(0, 3), np.random.uniform(-0.05, 0.05),
                    np.random.rand(7), False)
    loss = train_step(q_net, target_net, optimizer, loss_fn, buffer, batch_size=32)
    assert loss is not None
    assert np.isfinite(loss)


# ---------- update_target_network ----------

def test_update_target_network_copies_weights():
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size=7, n_actions=3)

    # nudge q_net's weights so it differs from target_net
    with torch.no_grad():
        for param in q_net.parameters():
            param.add_(1.0)

    update_target_network(q_net, target_net)

    for p_q, p_target in zip(q_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_q, p_target)


# ---------- build_agent ----------

def test_build_agent_networks_start_identical():
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size=7, n_actions=3)
    for p_q, p_target in zip(q_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_q, p_target)


def test_build_agent_target_network_is_frozen():
    q_net, target_net, optimizer, loss_fn = build_agent(obs_size=7, n_actions=3)
    assert not target_net.training  # eval() was called