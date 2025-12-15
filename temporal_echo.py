import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from collections import deque

@dataclass
class Config:
    # Environment
    room_size: float = 20.0
    agent_radius: float = 0.8
    button_pos: Tuple[float, float] = (4.0, 16.0)
    goal_pos: Tuple[float, float] = (16.0, 4.0)
    barrier_x: float = 10.0  # Barrier acts as a wall at x=10
    
    # Training
    device: str = "cpu"
    lr: float = 3e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    
    # Simulation
    max_steps: int = 100
    cycles_per_epoch: int = 10
    
    seed: int = 42

    def __post_init__(self):
        if torch.cuda.is_available(): self.device = "cuda"
        elif torch.backends.mps.is_available(): self.device = "cpu"

cfg = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class TemporalEnv:
    """
    A custom environment supporting Time Loops.
    State: [agent_x, agent_y, ghost_x, ghost_y, button_pressed, barrier_open]
    Action: [velocity_x, velocity_y] (continuous)
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = None
        self.steps = 0
        self.ghost_trajectory: List[Tuple[float, float]] = []
        self.history: List[Tuple[float, float]] = []
        self.has_ghost = False

    def reset(self, ghost_trajectory: Optional[List[Tuple[float, float]]] = None):
        self.steps = 0
        self.agent_pos = np.array([2.0, 2.0], dtype=np.float32)
        self.history = []
        
        if ghost_trajectory is not None and len(ghost_trajectory) > 0:
            self.ghost_trajectory = ghost_trajectory
            self.has_ghost = True
            self.ghost_pos = np.array(self.ghost_trajectory[0], dtype=np.float32)
        else:
            self.ghost_trajectory = []
            self.has_ghost = False
            self.ghost_pos = np.array([-5.0, -5.0], dtype=np.float32)

        return self._get_obs()

    def _get_obs(self):
        # Normalize observations to [-1, 1] range roughly for neural net stability
        s = self.cfg.room_size
        return np.array([
            self.agent_pos[0] / s,
            self.agent_pos[1] / s,
            self.ghost_pos[0] / s,
            self.ghost_pos[1] / s,
            1.0 if self._is_button_pressed() else 0.0,
            1.0 if self._is_barrier_open() else 0.0
        ], dtype=np.float32)

    def _is_button_pressed(self):
        d_agent = np.linalg.norm(self.agent_pos - np.array(self.cfg.button_pos))
        d_ghost = np.linalg.norm(self.ghost_pos - np.array(self.cfg.button_pos))
        radius = 2.0
        return d_agent < radius or d_ghost < radius

    def _is_barrier_open(self):
        return self._is_button_pressed()

    def step(self, action: np.ndarray):
        self.steps += 1
        
        # 1. Update Ghost Position
        if self.has_ghost:
            idx = min(self.steps, len(self.ghost_trajectory) - 1)
            self.ghost_pos = np.array(self.ghost_trajectory[idx])
        
        # 2. Apply Action
        # Action is velocity [-1, 1] -> scale to speed
        vel = np.clip(action, -1.0, 1.0) * 1.5 
        new_pos = self.agent_pos + vel

        # 3. Collision Logic
        # Bounds
        new_pos = np.clip(new_pos, 0, self.cfg.room_size)
        
        # Barrier Logic
        # Barrier is at x=10. If not open, you cannot cross x=10.
        barrier_open = self._is_barrier_open()
        
        # Check if we are trying to cross the barrier line
        crossed_barrier = (self.agent_pos[0] <= self.cfg.barrier_x and new_pos[0] > self.cfg.barrier_x) or \
                          (self.agent_pos[0] >= self.cfg.barrier_x and new_pos[0] < self.cfg.barrier_x)
        
        if crossed_barrier and not barrier_open:
            # Hit the wall, slide along y
            new_pos[0] = self.agent_pos[0] 
        
        self.agent_pos = new_pos
        self.history.append(tuple(self.agent_pos))

        # 4. Rewards
        reward = -0.01 # Time penalty
        done = False

        # Distance to goal reward
        dist_to_goal = np.linalg.norm(self.agent_pos - np.array(self.cfg.goal_pos))
        reward += (1.0 - dist_to_goal / self.cfg.room_size) * 0.1

        # Button reward (incentivize pressing button if goal is far/blocked)
        if self._is_button_pressed():
            reward += 0.05
        
        # Goal Achievement
        if dist_to_goal < 1.5:
            reward += 10.0
            done = True
        
        if self.steps >= self.cfg.max_steps:
            done = True

        return self._get_obs(), reward, done, {}

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor: outputs mean of action distribution
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Actor: learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic: outputs value estimate
        self.critic = nn.Linear(hidden_dim, 1)
        
        for layer in [self.actor_mean, self.critic]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        features = self.net(x)
        return features

    def get_action(self, x):
        features = self.forward(x)
        mean = self.actor_mean(features)
        std = self.actor_log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, self.critic(features), dist

    def get_value(self, x):
        return self.critic(self.forward(x))

class PPOAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = ActorCritic(obs_dim=6, action_dim=2, hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if not self.memory: return 0.0
        
        states, actions, old_log_probs, rewards, dones, values = zip(*self.memory)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.cfg.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.cfg.device)
        values = torch.cat(values).squeeze()

        advantages = []
        gae = 0
        with torch.no_grad():
            next_value = 0
            for i in reversed(range(len(rewards))):
                mask = 1.0 - dones[i]
                delta = rewards[i] + self.cfg.gamma * next_value * mask - values[i]
                gae = delta + self.cfg.gamma * 0.95 * mask * gae
                advantages.insert(0, gae)
                next_value = values[i]
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)
        returns = advantages + values

        total_loss = 0
        for _ in range(4):
            features = self.model(states)
            mean = self.model.actor_mean(features)
            std = self.model.actor_log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
            
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            new_values = self.model.critic(features).squeeze()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon, 1.0 + self.cfg.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(new_values, returns)
            
            loss = actor_loss + self.cfg.value_loss_coef * critic_loss - self.cfg.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            
        self.memory = []
        return total_loss

def render_env(env: TemporalEnv, container):
    scale = 20
    canvas_size = int(env.cfg.room_size * scale)

    def to_px(x, y):
        return int(x * scale), int(canvas_size - y * scale)

    ax, ay = to_px(*env.agent_pos)
    gx, gy = to_px(*env.ghost_pos)
    bx, by = to_px(*env.cfg.button_pos)
    glx, gly = to_px(*env.cfg.goal_pos)

    barrier_x_px = int(env.cfg.barrier_x * scale)

    agent_col = "#3498db" # Blue
    ghost_col = "rgba(100, 100, 100, 0.5)" # Translucent Grey
    button_col = "#e74c3c" if not env._is_button_pressed() else "#2ecc71" # Red -> Green
    barrier_col = "#ffffff" if not env._is_barrier_open() else "rgba(255,255,255, 0.1)"
    goal_col = "#f1c40f" # Gold

    svg = f"""
    <svg width="{canvas_size}" height="{canvas_size}" style="background-color: #2c3e50; border-radius: 8px;">
        <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />

        <line x1="{barrier_x_px}" y1="0" x2="{barrier_x_px}" y2="{canvas_size}" 
              stroke="{barrier_col}" stroke-width="6" stroke-dasharray="{'none' if not env._is_barrier_open() else '10,10'}" />

        <circle cx="{bx}" cy="{by}" r="15" fill="{button_col}" stroke="white" stroke-width="2" />
        <text x="{bx}" y="{by+35}" fill="white" font-family="monospace" font-size="12" text-anchor="middle">TIME LOCK</text>

        <circle cx="{glx}" cy="{gly}" r="20" fill="{goal_col}" stroke="white" stroke-width="2">
            <animate attributeName="r" values="20;25;20" dur="2s" repeatCount="indefinite" />
        </circle>
        <text x="{glx}" y="{gly+40}" fill="{goal_col}" font-family="monospace" font-size="14" text-anchor="middle" font-weight="bold">EXIT</text>

        <circle cx="{gx}" cy="{gy}" r="12" fill="{ghost_col}" stroke="#bdc3c7" stroke-width="1" stroke-dasharray="2,2" />
        <text x="{gx}" y="{gy-20}" fill="#bdc3c7" font-family="monospace" font-size="10" text-anchor="middle">ECHO</text>

        <circle cx="{ax}" cy="{ay}" r="12" fill="{agent_col}" stroke="white" stroke-width="2" />
        <text x="{ax}" y="{ay-20}" fill="white" font-family="monospace" font-size="10" text-anchor="middle">YOU</text>
        
    </svg>
    """
    container.markdown(svg, unsafe_allow_html=True)

def generate_agent_thought(value: float, button_pressed: bool, barrier_open: bool, loop_phase: int):
    if loop_phase == 1:
        if value < 0: return "Can't reach the exit. The wall is blocking me."
        if not button_pressed: return "Maybe I should press that button to help my future self?"
        return "Holding the button. Hope the echo works in the next loop."
    else:
        if not barrier_open: return "Waiting for my past self to press the button..."
        if barrier_open: return "The wall is gone! Thanks, past me. Running to exit!"
        return "Syncing with timeline..."

def main():
    st.set_page_config(page_title="Project Temporal Echo", layout="wide", page_icon="⏳")
    
    st.title("⏳ Temporal Echo: Deep RL with Time Loops")
    st.markdown("""
    **Concept:** An AI agent needs to escape a room. The door is locked by a button. 
    The catch? The button must be held to keep the door open, but the agent is alone.
    
    **Solution:** The agent enters a **Time Loop**.
    1.  **Loop 1 (The Sacrifice):** Agent learns to press the button, even though it can't exit.
    2.  **Loop 2 (The Escape):** The agent plays alongside a "Ghost" recording of Loop 1. The Ghost holds the button, the Agent escapes.
    """)
    
    with st.sidebar:
        st.header("Hyperparameters")
        train_speed = st.slider("Training Speed (Cycles/Step)", 1, 50, 10)
        lr_input = st.selectbox("Learning Rate", [1e-3, 3e-4, 1e-4], index=1)
        st.caption("Lower LR = More stable, slower learning")
        
        if st.button("Reset Model"):
            st.session_state.agent = PPOAgent(cfg)
            st.session_state.rewards_history = []
            st.session_state.epoch = 0
            st.rerun()
            
    if 'agent' not in st.session_state:
        set_seed(cfg.seed)
        st.session_state.agent = PPOAgent(cfg)
        st.session_state.rewards_history = []
        st.session_state.epoch = 0
        st.session_state.best_ghost = []

    agent = st.session_state.agent
    env = TemporalEnv(cfg)
    
    col_viz, col_stats = st.columns([1.5, 1])

    with col_viz:
        viz_container = st.empty()
        thought_bubble = st.empty()
        
    with col_stats:
        st.subheader("Neural Network Metrics")
        chart_rewards = st.empty()
        status_text = st.empty()
        st.markdown("---")
        st.markdown("**Live Logic Viz:**")
        logic_col1, logic_col2 = st.columns(2)
        metric_btn = logic_col1.empty()
        metric_door = logic_col2.empty()

    start_btn = st.button("▶ Start Time Loop Simulation")

    if start_btn:
        progress_bar = st.progress(0)
        
        for cycle in range(train_speed): 
            obs = env.reset(ghost_trajectory=[])
            done = False
            traj_1 = []
            ep_reward_1 = 0
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(cfg.device).unsqueeze(0)
                action, log_prob, val, _ = agent.model.get_action(obs_t)
                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)
                agent.store((obs_t, action, log_prob, reward, done, val))
                
                traj_1.append(tuple(env.agent_pos))
                ep_reward_1 += reward
                obs = next_obs
            loss_1 = agent.update()
            did_press = any([np.linalg.norm(np.array(p) - np.array(cfg.button_pos)) < 2.0 for p in traj_1])
            
            if did_press:
                active_ghost = traj_1
            else:
                active_ghost = [] # Ghost failed, so we are alone again

            obs = env.reset(ghost_trajectory=active_ghost)
            done = False
            ep_reward_2 = 0
            render_this_cycle = (cycle == train_speed - 1)
            
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).to(cfg.device).unsqueeze(0)
                action, log_prob, val, _ = agent.model.get_action(obs_t)
                
                act_np = action.cpu().numpy()[0]
                next_obs, reward, done, _ = env.step(act_np)
                
                agent.store((obs_t, action, log_prob, reward, done, val))
                ep_reward_2 += reward
                obs = next_obs

                if render_this_cycle:
                    render_env(env, viz_container)

                    metric_btn.metric("Button Status", "PRESSED" if env._is_button_pressed() else "Released", 
                                      delta_color="normal" if env._is_button_pressed() else "off")
                    metric_door.metric("Door Status", "OPEN" if env._is_barrier_open() else "LOCKED",
                                       delta="GO!" if env._is_barrier_open() else None)
                    
                    # Thoughts
                    t_text = generate_agent_thought(val.item(), env._is_button_pressed(), env._is_barrier_open(), 2 if active_ghost else 1)
                    thought_bubble.info(f"🧠 Agent Thought: *\"{t_text}\"*")
                    
                    time.sleep(0.02)

            loss_2 = agent.update()

            total_score = ep_reward_1 + ep_reward_2
            st.session_state.rewards_history.append(total_score)
            st.session_state.epoch += 1
            progress_bar.progress((cycle + 1) / train_speed)

        if len(st.session_state.rewards_history) > 0:
            rh = st.session_state.rewards_history
            # Moving Average
            ma = np.convolve(rh, np.ones(10)/10, mode='valid')
            chart_rewards.line_chart(ma if len(ma) > 0 else rh)
            
        status_text.markdown(f"""
        **Status Report:**
        - Epoch: `{st.session_state.epoch}`
        - Last Total Reward: `{total_score:.2f}`
        - Ghost Active: `{'YES' if did_press else 'NO (Loop 1 failed)'}`
        """)
        
        st.success("Simulation Batch Complete. Press Start to continue training.")

if __name__ == "__main__":
    main()
