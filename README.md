## ⏳ Temporal Echo: Deep RL with Time Loops

**Temporal Echo** is an experimental reinforcement-learning project that studies cooperative behavior across time: a single agent records a past trajectory (a "ghost") and then must cooperate with that ghost in a later timeline to solve a puzzle - pressing two buttons simultaneously to open a door.

This repository contains a Streamlit demo and a minimal LSTM + PPO implementation with several practical stability improvements (GAE, entropy bonus, advantage normalization) and a rasterized visualization for live animations.

---

## 💡 The Core Paradox

A classic coordination paradox: two buttons must be pressed simultaneously to open the exit, but only one agent exists. The only way to solve the puzzle is to *cooperate with your past self*: first record a trajectory where the agent presses Button A, then rewind time, spawn a ghost that replays that recorded trajectory, and in the second pass press Button B while the ghost holds Button A.

This setting yields a challenging sparse-reward RL problem with a temporal dependence between episodes.

---

## 🧠 Solution: Temporal Self-Cooperation

The reference implementation uses:

* **Policy**: RecurrentActorCritic (fully connected body → LSTMCell) producing Gaussian continuous actions.
* **Algorithm**: PPO with Generalized Advantage Estimation (GAE), advantage normalization, entropy bonus, and multiple PPO epochs per collected trajectory.
* **Training loop**: two-phase loop per epoch - (1) record a trajectory (Timeline 1), (2) reset environment with the recorded ghost and act in Timeline 2. Both trajectories are used to update the policy.

Key practical improvements included:

* GAE (lower variance advantages).
* Entropy regularization to encourage exploration.
* Reward shaping (dense small bonuses for pressing buttons and an extra bonus for simultaneous presses) while preserving the large final reward for reaching the exit.
* Careful tensor shape handling and `.detach()` to avoid autograd leaks.

---

## 🚀 Getting Started

### Requirements

A `requirements.txt` file is provided in this repo. Recommended Python versions: **3.8 - 3.11**. GPU recommended for faster training but not required.

### Install

```bash
git clone https://github.com/Ant1pozitive/temporal-echo.git
cd temporal_echo

python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate       # Windows (PowerShell: .venv\Scripts\Activate.ps1)

pip install -r requirements.txt
```

If you want GPU acceleration, install a PyTorch build that matches your CUDA toolkit. Example (Linux):

```bash
# visit https://pytorch.org for the correct CUDA build selector
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Run the demo

```bash
streamlit run temporal_echo.py -- --web
```

This will open the Streamlit UI where you can reset the model and run training loops. Use the animation preview to observe the agent and the ghost.

---

## 📈 Training Progress

* The app exposes metrics for **Timeline 1 Reward** and **Timeline 2 Reward** and a small line chart with recent timeline rewards.
* Practical recommendations:

  * Quick experiments: `train_iterations = 25`, `max_steps = 120`, `ppo_epochs = 6`, `entropy_coef = 0.02`.
  * Serious training: `train_iterations = 200`, `max_steps = 160`, `ppo_epochs = 8`, run on GPU when possible.

---

## ⚙️ Model Architecture

**RecurrentActorCritic** (implemented with PyTorch) - summary:

* Input: 6-dimensional observation vector [`agent_x`, `agent_y`, `ghost_x`, `ghost_y`, `buttonA_pressed`, `buttonB_pressed`] normalized to [0, 1] by room size.
* Body: Fully connected `Linear(obs_dim -> hidden)` + `Tanh`.
* Memory: `LSTMCell(hidden -> hidden)` to handle temporal dependence within episodes.
* Actor head: `Linear(hidden -> action_dim)` producing mean of Gaussian. Learned `log_std` parameter produces per-action variance.
* Critic head: `Linear(hidden -> 1)` producing value estimate.

### Extensions & Notes

* The design is intentionally simple so it is easy to replace the body with a Transformer encoder that attends over a (short) window of ghost states. If you want to try that, replace the `fc_body` with a Transformer encoder or cross-attention block that consumes a compact representation of the ghost trajectory.
* For larger-scale training, vectorized environments (multiple parallel rollouts) are strongly recommended.

---

## Troubleshooting & Tips

* **Frontend JS error `First argument must be a String, HTMLElement...`**: This appears when embedding complex inline SVG repeatedly via `st.markdown` - the demo switches to PNG frames rendered via Matplotlib and uses `st.image` to avoid this error.
* **No learning / negative rewards**: increase `train_iterations`, `max_steps`, or `ppo_epochs`; add entropy (`entropy_coef`) to encourage exploration; use the reward shaping suggested in the code. Consider curriculum training: first remove barrier or set the ghost to a trivial behavior so agent can learn basics.
* **Reproducibility**: set the `seed` in `Config` and ensure PyTorch, NumPy, and random seeds are set at startup. For deterministic GPU behavior, set additional PyTorch flags (see PyTorch docs).

---

## Next Steps (suggested experiments)

1. Vectorized rollout (`n_envs >= 8`) for higher data throughput.
2. Replace `fc_body` with an attention module that receives a compressed ghost-trajectory embedding.
3. Curriculum: train first without barrier, then gradually add barrier and increase required coordination.
4. Save checkpoints & export best policy. Add a button in UI to download a GIF/MP4 of final episodes.

---

## License

This project is open-source and available under the MIT License.
