# ⏳ Temporal Echo: Deep RL with Time Loops

A unique reinforcement learning project demonstrating **Temporal Self-Cooperation** - where a single agent learns to cooperate with a "Ghost" of its past self to solve an unsolvable physical paradox.

This project is implemented entirely in Python and uses a custom **Proximal Policy Optimization (PPO)** algorithm within a bespoke 2D physics environment visualized by Streamlit.

## 💡 The Core Paradox

The environment, known as the "Paradox Room," presents a classical problem:

* **Goal:** Reach the EXIT on the right side of the room.
* **Obstacle:** A barrier separates the Agent from the EXIT.
* **Solution:** A button must be *pressed and held* to open the barrier.

Since the Agent is alone, it cannot press the button and pass through the opened door simultaneously.

## 🧠 Solution: Temporal Self-Cooperation

The Agent learns to solve this by entering a two-phase **Time Loop**:

1.  **Phase 1 (The Sacrifice):** The Agent, knowing the goal is unreachable, learns the most beneficial action for the *system* - to run to the button and hold it until the episode ends. This trajectory is recorded.
2.  **Phase 2 (The Escape):** The recorded trajectory from Phase 1 is spawned as a translucent **"Echo" (Ghost)**. The Ghost runs and holds the button, opening the barrier for the Agent in the present loop to run straight to the EXIT.

The PPO algorithm is trained on the combined reward of both phases, forcing the policy to find a solution that prioritizes long-term, multi-temporal planning over immediate gain.

## 🛠️ Key Technical Features

* **Custom PPO Implementation:** The project includes a full, native implementation of the PPO algorithm (including GAE and Clipped Surrogate Objective) using PyTorch, without relying on external RL frameworks like Stable Baselines.
* **Continuous Control:** The Agent uses a Gaussian Policy (Normal Distribution) to output continuous velocity vectors (Ax, Ay), making the control challenging and precise.
* **Lightweight Physics Engine:** The environment uses NumPy for efficient 2D collision and movement logic.
* **State-of-the-Art Visualization (Streamlit):** The interactive UI provides real-time rendering using SVG for high-performance visuals on CPU, along with dynamic metrics and an **Agent Thought Bubble** that displays the model's inner monologue and planning status.

## 🚀 Getting Started

### Prerequisites

You need Python 3.8+ installed.

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Ant1pozitive/temporal-echo.git
cd temporal_echo
pip install -r requirements.txt
````

### Running the Simulation

Execute the Streamlit application from your terminal:

```bash
streamlit run temporal_echo.py -- --web
```

The application will open in your browser, where you can start the simulation and watch the Agent evolve its strategy in real-time.

## 📈 Training Progress

The Agent's learning curve is characterized by two major jumps:

1.  **Initial Spike:** Agent learns basic movement and navigation toward the button.
2.  **Second Spike (The Breakthrough):** Agent successfully stabilizes a policy where Phase 1 hits the button, and Phase 2 utilizes the generated Ghost to reach the exit, leading to high, consistent rewards.

## ⚙️ Model Architecture

The policy is an Actor-Critic architecture:

  * **Input State (6D Vector):** `[Agent_X, Agent_Y, Ghost_X, Ghost_Y, Button_Pressed, Barrier_Open]`
  * **Shared Net:** 2x fully connected layers with Tanh activation.
  * **Actor (Policy):** Outputs the mean and standard deviation for the 2D action vector (velocity).
  * **Critic (Value):** Estimates the expected cumulative reward (Value function).

## License

This project is open-source and available under the MIT License.
