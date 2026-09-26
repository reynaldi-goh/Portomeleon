# Iteration 3: DQN Agent

## Goal

The goal of this iteration was to adapt the DQN agent from the single stock prototype so it works with the multi stock trading environment built in Iteration 2. This mainly meant figuring out how one trained network could be used to evaluate several stocks at once each day, packaging its per stock decisions into the format the new environment expects, and fixing a trade-sizing issue that surfaced once real training runs were compared across hyperparameters.

## What Was Built

The agent is implemented in `src/agent.py`. Most of the prototype's code did not need to change. `QNetwork`, `ReplayBuffer`, `margin_to_size`, `train_step`, `update_target_network`, and `build_agent` all already operated on a single observation or a single transition at a time, so none of them care how many stocks exist in the portfolio. The only change needed to these was `obs_size`, now 7 instead of 5, matching the two portfolio context features added to the observation vector in Iteration 2.

The first new addition is `select_portfolio_actions()`. It takes the dictionary of observations returned by the environment's `_get_obs()`, one vector per ticker, and calls the original `select_action_and_size()` once per ticker using the same network. It returns a dictionary of `{ticker: (action, size_fraction)}`, which is exactly the format `TradingEnv.step()` expects. The network itself was never made aware that multiple stocks exist. It still only ever answers one question, given this one stock's state, what should I do, and how confident am I. The looping across stocks and the decision of how much cash goes where both happen outside the network, in the environment and in this wrapper function.

The second new addition is `MarginScaler`, along with an optional `scaler` argument on `select_action_and_size()`. Instead of turning a raw Q-value margin into a trade size with a fixed scale (the original `margin_to_size`, still kept as the default fallback), `MarginScaler` ranks a margin against a rolling window of recent margins the model has actually produced, and converts that rank directly into a 0 to 1 confidence. `get_margin_stats()` and `reset_margin_log()` were also added to summarize the margins and sizes seen during a run, for debugging.

A demo script was also built to run the whole pipeline end to end, from Iteration 1's data layer, through a `load_multi_stock_data()` helper that fetches several tickers and aligns them to the same trading dates, into the Iteration 2 environment, and finally training and stepping through the Iteration 3 agent.

## Design Decisions

Wrapping the network instead of changing it: the alternative to `select_portfolio_actions()` would have been to make `QNetwork` itself aware of multiple stocks, for example by widening its input to take every ticker's state at once. This was rejected because the network's job, judging one state and producing one confidence, does not actually change when there are more stocks to evaluate. Keeping the network single-stock also means it can be reused unchanged from the prototype, and the portfolio-level logic stays isolated in the environment and in this one wrapper function rather than spreading into the model itself.

Confidence ranked against recent margins instead of a fixed scale: `margin_to_size` divides a raw margin by a fixed constant (0.02) to get a confidence. In practice, how large a typical margin is depends on the model itself, for example it shifted noticeably across different alpha values in the hyperparameter grid, so a fixed scale meant trade sizes could saturate at the maximum for one trained model and barely move for another, even when both were equally confident relative to their own typical margins. `MarginScaler` fixes this by comparing a margin against a rolling window of the margins that specific model has actually been producing, so a size of, say, 0.7 confidence means the same thing (a comparatively strong decision) regardless of the model's own margin scale. `freeze()` is used once training ends, so a saved model keeps sizing trades against the margin distribution it ended training with, rather than continuing to shift as it is evaluated.

## Testing

Unit tests were written in `tests/test_agent.py` using pytest, covering the network, the replay buffer, the margin scaler, action selection, and the training step.

- `QNetwork` tests confirm the output shape matches the number of actions, and that a forward pass never produces NaN or infinite values.
- `ReplayBuffer` tests confirm that pushing an experience increases its length, that it correctly drops the oldest experiences once it reaches capacity, and that a sampled batch has the expected shapes.
- `MarginScaler` tests confirm that confidence falls back to a neutral 0.5 before enough margins have been seen, that confidence correctly ranks a new margin against stored history, that `freeze()` actually stops further updates from changing anything, and that its internal count never grows past its capacity.
- `margin_to_size` tests confirm the two boundary cases, a margin of zero maps to the minimum size, and a very large margin gets capped at the maximum size, along with a general check that the result always stays within bounds across a range of margins.
- `select_action_and_size` tests force the network's weights directly so a specific action is guaranteed to win, rather than relying on chance. This confirms that a forced hold decision returns a size fraction of zero, that a forced buy decision returns a positive size fraction within the expected range, and that full exploration (epsilon equal to 1) always returns a valid action and size. A further test confirms that passing a `MarginScaler` in produces a size based on its ranked confidence rather than the fixed scale.
- `select_portfolio_actions` tests confirm that every ticker passed in receives an entry in the returned dictionary, and that each entry contains a valid action and size.
- `train_step` tests confirm that it correctly returns nothing when the replay buffer does not yet have enough experiences for a full batch, and that once it does, the returned loss is a finite number.
- `update_target_network` and `build_agent` tests confirm that the target network starts with identical weights to the main network, that it is correctly frozen in evaluation mode, and that copying weights across after training actually updates it.

All 23 tests currently pass.