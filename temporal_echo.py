import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class Config:
    # Physics / World
    room_size: float = 20.0
    agent_radius: float = 0.8
    button_a_pos: np.ndarray = field(default_factory=lambda: np.array([5.0, 15.0]))
    button_b_pos: np.ndarray = field(default_factory=lambda: np.array([15.0, 15.0]))
    goal_pos: np.ndarray = field(default_factory=lambda: np.array([10.0, 2.0]))
    barrier_y: float = 8.0
    
    # Training
    device: str = "cpu"
    lr: float = 0.0003
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Model Architecture
    hidden_dim: int = 64
    
    # Simulation
    max_steps: int = 80
    train_iterations: int = 5
    seed: int = 42

    def __post_init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"

cfg = Config()

class TemporalEnv:
    """
    Environment with 'Time Travel' mechanics.
    Goal: Open the door by pressing Button A and Button B simultaneously.
    Problem: There is only one agent.
    Solution: Cooperative play with a recorded 'Ghost' from a previous timeline.
    """
    def __init__(self, config: Config):
        self.cfg = config
        self.steps = 0
        self.agent_pos = np.array([10.0, 18.0], dtype=np.float32)
        self.ghost_pos = np.array([-10.0, -10.0], dtype=np.float32)
        self.ghost_traj = []
        self.has_ghost = False
        self.ghost_idx = 0

    def reset(self, ghost_trajectory: List[np.ndarray] = None):
        self.steps = 0
        self.agent_pos = np.array([10.0, 18.0], dtype=np.float32)
        
        if ghost_trajectory is not None and len(ghost_trajectory) > 0:
            self.ghost_traj = ghost_trajectory
            self.has_ghost = True
            self.ghost_idx = 0
            self.ghost_pos = self.ghost_traj[0]
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
                self.ghost_pos = self.ghost_traj[self.ghost_idx]
                self.ghost_idx += 1
            else:
                self.ghost_pos = self.ghost_traj[-1]

        # Action is [vel_x, vel_y] clamped to [-1, 1]
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
        
        dist_to_goal = np.linalg.norm(self.agent_pos - self.cfg.goal_pos)

        if self._is_pressed(self.cfg.button_a_pos): reward += 0.02
        if self._is_pressed(self.cfg.button_b_pos): reward += 0.02

        if dist_to_goal < 1.5:
            reward += 10.0
            done = True
        
        if self.steps >= self.cfg.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {}

# MODEL: RECURRENT PPO (LSTM)
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
        x = self.fc_body(x)
        hx_new, cx_new = self.lstm(x, (hx, cx))
        return hx_new, cx_new

    def get_action(self, x, hx, cx):
        h_out, c_out = self.forward(x, hx, cx)
        
        mean = self.actor(h_out)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(h_out)
        
        return action, log_prob, value, h_out, c_out

class PPOAgent:
    def __init__(self, config: Config):
        self.cfg = config
        self.model = RecurrentActorCritic(6, 2, config.hidden_dim).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.memory = []
    
    def store(self, transition):
        # transition: (obs, h, c, action, log_prob, reward, done)
        self.memory.append(transition)

    def finish_episode(self):
        if not self.memory: return 0.0
        obs_lst, h_lst, c_lst, act_lst, lp_lst, rew_lst, done_lst = zip(*self.memory)
        
        obs = torch.stack(obs_lst).to(self.cfg.device)
        h_ins = torch.stack(h_lst).detach().to(self.cfg.device).squeeze(1)
        c_ins = torch.stack(c_lst).detach().to(self.cfg.device).squeeze(1)
        actions = torch.stack(act_lst).to(self.cfg.device)
        old_log_probs = torch.stack(lp_lst).to(self.cfg.device)
        rewards = torch.tensor(rew_lst, dtype=torch.float32).to(self.cfg.device)
        
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.cfg.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        total_loss = 0
        for _ in range(4):
            h, c = h_ins[0].unsqueeze(0), c_ins[0].unsqueeze(0)

            curr_log_probs = []
            curr_values = []

            h_curr, c_curr = h_ins[0].unsqueeze(0), c_ins[0].unsqueeze(0)
            
            for t in range(len(obs)):
                h_curr, c_curr = self.model.forward(obs[t].unsqueeze(0), h_curr, c_curr)

                mean = self.model.actor(h_curr)
                std = self.model.log_std.exp().expand_as(mean)
                dist = Normal(mean, std)
                lp = dist.log_prob(actions[t]).sum(-1)
                val = self.model.critic(h_curr)
                
                curr_log_probs.append(lp)
                curr_values.append(val)
                
            curr_log_probs = torch.stack(curr_log_probs).squeeze()
            curr_values = torch.stack(curr_values).squeeze()

            ratio = torch.exp(curr_log_probs - old_log_probs)
            advantage = returns - curr_values.detach()
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * advantage
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(curr_values, returns)
            
            loss = actor_loss + self.cfg.value_coef * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            
        self.memory = []
        return total_loss

def render_svg(env: TemporalEnv, phase_text: str):
    s = 20
    W = int(env.cfg.room_size * s)
    H = int(env.cfg.room_size * s)
    
    def to_px(pos): return int(pos[0]*s), int(H - pos[1]*s)
    
    ax, ay = to_px(env.agent_pos)
    gx, gy = to_px(env.ghost_pos)
    bax, bay = to_px(env.cfg.button_a_pos)
    bbx, bby = to_px(env.cfg.button_b_pos)
    gox, goy = to_px(env.cfg.goal_pos)
    bar_y_px = int(H - env.cfg.barrier_y * s)

    c_bg = "#1e1e2e"
    c_btn_a = "#ff5555" if not env._is_pressed(env.cfg.button_a_pos) else "#50fa7b"
    c_btn_b = "#8be9fd" if not env._is_pressed(env.cfg.button_b_pos) else "#50fa7b"
    c_agent = "#ff79c6"
    c_ghost = "rgba(189, 147, 249, 0.4)"
    
    door_open = env._is_pressed(env.cfg.button_a_pos) and env._is_pressed(env.cfg.button_b_pos)
    door_stroke = "none" if door_open else "white"
    
    svg = f"""
    <svg width="{W}" height="{H}" style="background-color: {c_bg}; border-radius: 8px; border: 2px solid #6272a4;">
        <text x="10" y="25" fill="#6272a4" font-family="monospace" font-weight="bold">{phase_text}</text>

        <line x1="0" y1="{bar_y_px}" x2="{W}" y2="{bar_y_px}" stroke="{door_stroke}" stroke-width="6" stroke-dasharray="10, 5" />
        
        <circle cx="{gox}" cy="{goy}" r="25" fill="none" stroke="#f1fa8c" stroke-width="2" stroke-dasharray="4,2">
             <animateTransform attributeName="transform" type="rotate" from="0 {gox} {goy}" to="360 {gox} {goy}" dur="10s" repeatCount="indefinite"/>
        </circle>
        <text x="{gox}" y="{goy+5}" fill="#f1fa8c" text-anchor="middle" font-family="Arial" font-size="12">EXIT</text>

        <circle cx="{bax}" cy="{bay}" r="15" fill="{c_btn_a}" stroke="white" stroke-width="2"/>
        <text x="{bax}" y="{bay+5}" fill="black" text-anchor="middle" font-weight="bold" font-family="Arial" font-size="12">A</text>
        
        <circle cx="{bbx}" cy="{bby}" r="15" fill="{c_btn_b}" stroke="white" stroke-width="2"/>
        <text x="{bbx}" y="{bby+5}" fill="black" text-anchor="middle" font-weight="bold" font-family="Arial" font-size="12">B</text>

        <g>
            <circle cx="{gx}" cy="{gy}" r="12" fill="{c_ghost}" stroke="#bd93f9" stroke-width="1" stroke-dasharray="2,2" />
        </g>

        <circle cx="{ax}" cy="{ay}" r="12" fill="{c_agent}" stroke="white" stroke-width="2" />
        
        <line x1="{ax}" y1="{ay}" x2="{gx}" y2="{gy}" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
    </svg>
    """
    return svg

def main():
    st.set_page_config(page_title="Temporal Echo AI", layout="wide", page_icon="📼")
    
    # Custom CSS for "VHS" aesthetic
    st.markdown("""
    <style>
    .stApp { background-color: #0e0e11; color: #dcdcdc; }
    h1 { color: #ff79c6; text-shadow: 2px 2px #bd93f9; }
    </style>
    """, unsafe_allow_html=True)

    st.title("📼 Project Temporal Echo")
    st.markdown("""
    **The Puzzle:** To exit, **Button A** and **Button B** must be pressed *simultaneously*.
    **The Problem:** You are alone.
    **The Solution:** Record your actions in Timeline 1 (Ghost), then cooperate with your past self in Timeline 2.
    """)

    with st.sidebar:
        st.header("Control Panel")
        if st.button("Reset Brain (Clear Model)", type="primary"):
            st.session_state.agent = PPOAgent(cfg)
            st.session_state.history = []
            st.rerun()
        
        st.info("Architecture: LSTM-PPO\nContext: Temporal Self-Attention")

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

        screen.markdown(render_svg(env, "WAITING FOR START..."), unsafe_allow_html=True)

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
            
            # Agent tries to press Button A to help future self.
            # Ghost is inactive/empty here.
            
            obs = env.reset(ghost_trajectory=[])
            h = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            c = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            
            traj_1_pos = []
            r1_total = 0
            
            for t in range(cfg.max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)

                with torch.no_grad():
                    action, log_prob, _, h_new, c_new = agent.model.get_action(obs_t, h, c)

                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)

                agent.store((obs_t.squeeze(0), h, c, action.squeeze(0), log_prob, reward, done))

                traj_1_pos.append(env.agent_pos.copy())

                obs = next_obs
                h, c = h_new, c_new
                r1_total += reward
                
                if done: break
            
            loss1 = agent.finish_episode()

            if epoch % 2 == 0:
                overlay.info("⏪ REWINDING REALITY...")
                time.sleep(0.1)
                overlay.empty()

            # Agent cooperates with Ghost (traj_1_pos)
            obs = env.reset(ghost_trajectory=traj_1_pos)
            h = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            c = torch.zeros(1, cfg.hidden_dim).to(cfg.device)
            r2_total = 0
            
            for t in range(cfg.max_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(cfg.device)
                
                with torch.no_grad():
                    action, log_prob, _, h_new, c_new = agent.model.get_action(obs_t, h, c)
                
                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)
                
                agent.store((obs_t.squeeze(0), h, c, action.squeeze(0), log_prob, reward, done))
                
                obs = next_obs
                h, c = h_new, c_new
                r2_total += reward

                if epoch == cfg.train_iterations - 1:
                    svg = render_svg(env, f"TIMELINE 2 | STEP {t}")
                    screen.markdown(svg, unsafe_allow_html=True)
                    time.sleep(0.02)
                
                if done: break

            loss2 = agent.finish_episode()

            st.session_state.history.append(r2_total)
            progress_bar.progress((epoch+1) / cfg.train_iterations)

        chart_loss.line_chart(st.session_state.history)
        m1.metric("Timeline 1 Reward", f"{r1_total:.2f}")
        m2.metric("Timeline 2 Reward", f"{r2_total:.2f}", delta="SOLVED" if r2_total > 5.0 else None)
        
        st.success("Temporal Convergence Complete. Agent memory updated.")

if __name__ == "__main__":
    main()
