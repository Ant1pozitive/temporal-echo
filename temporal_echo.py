import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from io import BytesIO
from PIL import Image

PIXELS_PER_UNIT = 20

@dataclass
class Config:
    # Physics / World
    room_size: float = 20.0
    agent_radius: float = 0.8
    button_a_pos: np.ndarray = field(default_factory=lambda: np.array([5.0, 15.0]))
    button_b_pos: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0]))
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([10.0, 2.0]))
    barrier_y: float = 8.0

    # Training / PPO
    device: str = "cpu"
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    ppo_epochs: int = 6

    # Model Architecture
    hidden_dim: int = 64

    # Simulation
    max_steps: int = 120
    train_iterations: int = 25
    seed: int = 42

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"

cfg = Config()

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
if cfg.device == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)

class TemporalEnv:
    def __init__(self, config: Config):
        self.cfg = config
        self.steps = 0
        self.agent_pos = np.array([10.0, 18.0], dtype=np.float32)
        self.ghost_pos = np.array([-10.0, -10.0], dtype=np.float32)
        self.ghost_traj: List[np.ndarray] = []
        self.has_ghost = False
        self.ghost_idx = 0

    def reset(self, ghost_trajectory: List[np.ndarray] = None):
        self.steps = 0
        self.agent_pos = np.array([10.0, 18.0], dtype=np.float32)

        if ghost_trajectory is not None and len(ghost_trajectory) > 0:
            self.ghost_traj = ghost_trajectory
            self.has_ghost = True
            self.ghost_idx = 0
            self.ghost_pos = np.array(self.ghost_traj[0], dtype=np.float32)
        else:
            self.ghost_traj = []
            self.has_ghost = False
            self.ghost_pos = np.array([-10.0, -10.0], dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        s = self.cfg.room_size
        obs = np.array([
            self.agent_pos[0] / s, self.agent_pos[1] / s,
            self.ghost_pos[0] / s, self.ghost_pos[1] / s,
            1.0 if self._is_pressed(self.cfg.button_a_pos) else 0.0,
            1.0 if self._is_pressed(self.cfg.button_b_pos) else 0.0
        ], dtype=np.float32)
        return obs

    def _is_pressed(self, btn_pos):
        da = np.linalg.norm(self.agent_pos - btn_pos)
        dg = np.linalg.norm(self.ghost_pos - btn_pos)
        return da < 1.5 or dg < 1.5

    def step(self, action):
        self.steps += 1
        if self.has_ghost:
            if self.ghost_idx < len(self.ghost_traj):
                self.ghost_pos = np.array(self.ghost_traj[self.ghost_idx], dtype=np.float32)
                self.ghost_idx += 1
            else:
                self.ghost_pos = np.array(self.ghost_traj[-1], dtype=np.float32)

        velocity = np.clip(action, -1.0, 1.0) * 1.2
        new_pos = self.agent_pos + velocity
        new_pos = np.clip(new_pos, 0, self.cfg.room_size)

        door_open = self._is_pressed(self.cfg.button_a_pos) and self._is_pressed(self.cfg.button_b_pos)
        if not door_open:
            crossing_down = self.agent_pos[1] >= self.cfg.barrier_y and new_pos[1] < self.cfg.barrier_y
            crossing_up = self.agent_pos[1] <= self.cfg.barrier_y and new_pos[1] > self.cfg.barrier_y
            if crossing_down or crossing_up:
                new_pos[1] = self.agent_pos[1]

        self.agent_pos = new_pos

        reward = -0.01
        done = False

        if self._is_pressed(self.cfg.button_a_pos):
            reward += 0.02
        if self._is_pressed(self.cfg.button_b_pos):
            reward += 0.02

        if self._is_pressed(self.cfg.button_a_pos) and self._is_pressed(self.cfg.button_b_pos):
            reward += 1.0

        dist_to_goal = np.linalg.norm(self.agent_pos - self.cfg.goal_pos)
        if dist_to_goal < 1.5:
            reward += 10.0
            done = True

        if self.steps >= self.cfg.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

# Model: Recurrent Actor-Critic (LSTM)
class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh()
        )
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, x, hx, cx):
        # x: (batch=1, obs_dim)
        x = self.fc_body(x)
        hx_new, cx_new = self.lstm(x, (hx, cx))
        return hx_new, cx_new

    def get_action(self, x, hx, cx):
        h_out, c_out = self.forward(x, hx, cx)
        mean = self.actor(h_out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # shape (batch,)
        value = self.critic(h_out)  # shape (batch, 1)
        return action, log_prob, value, h_out, c_out

# PPO Agent (with GAE + entropy)
class PPOAgent:
    def __init__(self, config: Config):
        self.cfg = config
        self.model = RecurrentActorCritic(6, 2, config.hidden_dim).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def finish_episode(self):
        if not self.memory:
            return 0.0

        obs_lst, h_lst, c_lst, act_lst, lp_lst, rew_lst, done_lst = zip(*self.memory)
        T = len(obs_lst)

        obs = torch.stack(obs_lst).to(self.cfg.device)            # (T, obs_dim)
        h_ins = torch.stack(h_lst).detach().to(self.cfg.device).squeeze(1)  # (T, hidden_dim)
        c_ins = torch.stack(c_lst).detach().to(self.cfg.device).squeeze(1)  # (T, hidden_dim)
        actions = torch.stack(act_lst).to(self.cfg.device)        # (T, act_dim)
        old_log_probs = torch.stack(lp_lst).to(self.cfg.device).squeeze()  # (T,)
        rewards = torch.tensor(rew_lst, dtype=torch.float32).to(self.cfg.device)  # (T,)
        dones = torch.tensor(done_lst, dtype=torch.float32).to(self.cfg.device)    # (T,)

        # Recompute values & log_probs & entropies along trajectory (sequence forward)
        with torch.no_grad():
            h = h_ins[0].unsqueeze(0).clone()
            c = c_ins[0].unsqueeze(0).clone()
            values = []
            for t in range(T):
                h, c = self.model.forward(obs[t].unsqueeze(0), h, c)  # h: (1, H)
                v = self.model.critic(h).squeeze(0)  # (1,) -> squeeze -> ()
                values.append(v)
            values = torch.stack(values).squeeze()  # (T,)

        values_np = values.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        dones_np = dones.cpu().numpy()

        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones_np[t]
                nextvalues = 0.0
            else:
                nextnonterminal = 1.0 - dones_np[t+1]
                nextvalues = values_np[t+1]
            delta = rewards_np[t] + self.cfg.gamma * nextvalues * nextnonterminal - values_np[t]
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.cfg.ppo_epochs):
            h = h_ins[0].unsqueeze(0).clone()
            c = c_ins[0].unsqueeze(0).clone()

            curr_log_probs = []
            curr_values = []
            curr_entropies = []

            for t in range(T):
                h, c = self.model.forward(obs[t].unsqueeze(0), h, c)
                mean = self.model.actor(h)
                std = self.model.log_std.exp().expand_as(mean)
                dist = Normal(mean, std)

                lp = dist.log_prob(actions[t].unsqueeze(0)).sum(dim=-1)  # (1,)
                ent = dist.entropy().sum(dim=-1)  # (1,)
                val = self.model.critic(h).squeeze(0)  # (1,) -> ()

                curr_log_probs.append(lp.squeeze(0))
                curr_entropies.append(ent.squeeze(0))
                curr_values.append(val)

            curr_log_probs = torch.stack(curr_log_probs).to(self.cfg.device)   # (T,)
            curr_entropies = torch.stack(curr_entropies).to(self.cfg.device)   # (T,)
            curr_values = torch.stack(curr_values).to(self.cfg.device)         # (T,)

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(curr_values, returns)
            entropy_loss = -curr_entropies.mean()

            loss = actor_loss + self.cfg.value_coef * critic_loss + self.cfg.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        self.memory = []
        return total_loss

def render_frame_image(env: TemporalEnv, phase_text: str, px_per_unit: int = PIXELS_PER_UNIT) -> np.ndarray:
    room = env.cfg.room_size
    dpi = 100
    figsize = (6, 6)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")

    ax.set_xlim(0, room)
    ax.set_ylim(0, room)
    ax.set_aspect('equal')
    ax.axis('off')

    by = env.cfg.barrier_y
    ax.plot([0, room], [by, by], linewidth=3, color='white', linestyle=(0, (10, 5)))

    gx, gy = env.cfg.goal_pos
    exit_circle = Circle((gx, gy), 1.2, facecolor='none', edgecolor='#f1fa8c', linewidth=1.5, linestyle='--')
    ax.add_patch(exit_circle)
    ax.text(gx, gy, "EXIT", color='#f1fa8c', ha='center', va='center', fontsize=10, fontweight='bold')

    ba = env.cfg.button_a_pos
    bb = env.cfg.button_b_pos
    c_btn_a = "#50fa7b" if env._is_pressed(ba) else "#ff5555"
    c_btn_b = "#50fa7b" if env._is_pressed(bb) else "#8be9fd"
    ax.add_patch(Circle((ba[0], ba[1]), 0.6, color=c_btn_a, ec='white', linewidth=1))
    ax.text(ba[0], ba[1], "A", ha='center', va='center', fontweight='bold')
    ax.add_patch(Circle((bb[0], bb[1]), 0.6, color=c_btn_b, ec='white', linewidth=1))
    ax.text(bb[0], bb[1], "B", ha='center', va='center', fontweight='bold')

    gx_pos = float(env.ghost_pos[0])
    gy_pos = float(env.ghost_pos[1])
    ghost_patch = Circle((gx_pos, gy_pos), 0.5, facecolor=(189/255,147/255,249/255,0.4), edgecolor='#bd93f9', linewidth=1)
    ax.add_patch(ghost_patch)

    ax.add_patch(Circle((env.agent_pos[0], env.agent_pos[1]), 0.5, facecolor='#ff79c6', edgecolor='white', linewidth=1))

    ax.plot([env.agent_pos[0], gx_pos], [env.agent_pos[1], gy_pos], color=(1,1,1,0.08), linewidth=0.8)

    ax.text(0.5, room - 0.5, phase_text, fontsize=9, color='#6272a4', fontfamily='monospace', fontweight='bold')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    arr = np.array(img)
    plt.close(fig)
    return arr

def main():
    st.set_page_config(page_title="Temporal Echo AI", layout="wide", page_icon="📼")

    st.markdown("""
    <style>
    .stApp { background-color: #0e0e11; color: #dcdcdc; }
    h1 { color: #ff79c6; text-shadow: 2px 2px #bd93f9; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📼 Temporal Echo")
    st.markdown("""
    **The Puzzle:** To exit, **Button A** and **Button B** must be pressed *simultaneously*.\n
    **The Problem:** You are alone.\n
    **The Solution:** Record your actions in Timeline 1 (Ghost), then cooperate with your past self in Timeline 2.
    """)

    with st.sidebar:
        st.header("Control Panel")
        if st.button("Reset Brain (Clear Model)"):
            st.session_state.agent = PPOAgent(cfg)
            st.session_state.history = []
            st.rerun()
        st.info("Architecture: LSTM-PPO (GAE + Entropy)")

    if 'agent' not in st.session_state:
        st.session_state.agent = PPOAgent(cfg)
        st.session_state.history = []

    env = TemporalEnv(cfg)
    agent = st.session_state.agent

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Live Simulation")
        screen = st.empty()
        overlay = st.empty()
        img0 = render_frame_image(env, "WAITING FOR START...")
        screen.image(img0, use_column_width=True)

    with col2:
        st.subheader("Neural Training Metrics")
        chart_loss = st.empty()
        metrics_container = st.container()
        chart_loss.line_chart([0])
        with metrics_container:
            m1, m2 = st.columns(2)
            m1.metric("Timeline 1 Reward", "0.00")
            m2.metric("Timeline 2 Reward", "0.00")

    if st.button("▶ START TEMPORAL LOOP"):
        progress_bar = st.progress(0)

        for epoch in range(cfg.train_iterations):
            obs = env.reset(ghost_trajectory=[])
            h = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            c = torch.zeros(1, cfg.hidden_dim).to(cfg.device)

            traj_1_pos = []
            r1_total = 0.0

            for t in range(cfg.max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    action, log_prob, _, h_new, c_new = agent.model.get_action(obs_t, h, c)

                lp_to_store = log_prob.squeeze()
                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)

                agent.store((obs_t.squeeze(0).detach(), h.detach(), c.detach(), action.squeeze(0).detach(), lp_to_store.detach(), float(reward), bool(done)))
                traj_1_pos.append(env.agent_pos.copy())

                obs = next_obs
                h, c = h_new, c_new
                r1_total += float(reward)
                if done:
                    break

            loss1 = agent.finish_episode()

            if epoch % 2 == 0:
                overlay.info("⏪ REWINDING REALITY...")
                time.sleep(0.1)
                overlay.empty()

            obs = env.reset(ghost_trajectory=traj_1_pos)
            h = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            c = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            r2_total = 0.0

            for t in range(cfg.max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                with torch.no_grad():
                    action, log_prob, _, h_new, c_new = agent.model.get_action(obs_t, h, c)

                lp_to_store = log_prob.squeeze()
                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)

                agent.store((obs_t.squeeze(0).detach(), h.detach(), c.detach(), action.squeeze(0).detach(), lp_to_store.detach(), float(reward), bool(done)))

                obs = next_obs
                h, c = h_new, c_new
                r2_total += float(reward)

                if epoch == cfg.train_iterations - 1:
                    frame = render_frame_image(env, f"TIMELINE 2 | STEP {t}")
                    screen.image(frame, use_column_width=True)
                    time.sleep(0.02)

                if done:
                    break

            loss2 = agent.finish_episode()

            st.session_state.history.append(r2_total)
            percent = int((epoch + 1) / cfg.train_iterations * 100)
            progress_bar.progress(percent)

        chart_loss.line_chart(st.session_state.history)
        try:
            m1.metric("Timeline 1 Reward", f"{r1_total:.2f}")
            m2.metric("Timeline 2 Reward", f"{r2_total:.2f}", delta="SOLVED" if r2_total > 5.0 else None)
        except Exception:
            pass

        st.success("Temporal Convergence Complete. Agent memory updated.")

if __name__ == "__main__":
    main()
