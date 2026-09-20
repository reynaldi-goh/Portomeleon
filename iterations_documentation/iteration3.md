# Iteration 3: DQN Agent

## Goal

The goal of this iteration was to adapt the DQN agent from the single stock prototype so it works with the multi stock trading environment built in Iteration 2. This mainly meant figuring out how one trained network could be used to evaluate several stocks at once each day, and packaging its per stock decisions into the format the new environment expects.

## What Was Built

The agent is implemented in `src/agent.py`. Most of the prototype's code did not need to change. `QNetwork`, `ReplayBuffer`, `margin_to_size`, `train_step`, `update_target_network`, and `build_agent` all already operated on a single observation or a single transition at a time, so none of them care how many stocks exist in the portfolio. The only real change needed was to `obs_size`, which is now 7 instead of 5, matching the two portfolio context features added to the observation vector in Iteration 2.

The one new function added this iteration is `select_portfolio_actions()`. It takes the dictionary of observations returned by the environment's `_get_obs()`, one vector per ticker, and calls the original `select_action_and_size()` once per ticker using the same network. It then returns a dictionary of `{ticker: (action, size_fraction)}`, which is exactly the format `TradingEnv.step()` expects. In other words, the network itself was never made aware that multiple stocks exist. It still only ever answers one question, given this one stock's state, what should I do, and how confident am I. The looping across stocks and the decision of how much cash goes where both happen outside the network, in the environment and in this new wrapper function.

A demo script was also built to run the whole pipeline end to end, from Iteration 1's `load_price_data()`, through a new `load_multi_stock_data()` helper that fetches several tickers and aligns them to the same trading dates, into the Iteration 2 environment, and finally training and stepping through the Iteration 3 agent.

## Testing

Unit tests were written in `tests/test_agent.py` using pytest, covering the network, the replay buffer, the confidence scoring function, action selection, and the training step.

- `QNetwork` tests confirm the output shape matches the number of actions, and that a forward pass never produces NaN or infinite values.
- `ReplayBuffer` tests confirm that pushing an experience increases its length, that it correctly drops the oldest experiences once it reaches capacity, and that a sampled batch has the expected shapes.
- `margin_to_size` tests confirm the two boundary cases, a margin of zero maps to the minimum size, and a very large margin gets capped at the maximum size, along with a general check that the result always stays within bounds across a range of margins.
- `select_action_and_size` tests force the network's weights directly so a specific action is guaranteed to win, rather than relying on chance. This confirms that a forced hold decision returns a size fraction of zero, and that a forced buy decision returns a positive size fraction within the expected range. A separate test also confirms that full exploration, epsilon equal to 1, still always returns a valid action and size.
- `select_portfolio_actions` tests confirm that every ticker passed in receives an entry in the returned dictionary, and that each entry contains a valid action and size.
- `train_step` tests confirm that it correctly returns nothing when the replay buffer does not yet have enough experiences for a full batch, and that once it does, the returned loss is a finite number.
- `update_target_network` and `build_agent` tests confirm that the target network starts with identical weights to the main network, that it is correctly frozen in evaluation mode, and that copying weights across after training actually updates it.

All 15 tests currently pass.

```
tests\test_agent.py ...............                       [100%]
======================== 15 passed in 1.87s ========================
```

