[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Y9wdwI_n)

# 🚁 Autonomous Drone Landing System — Deep Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.3-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

</div>

---

## 🎓 Academic Information

| Field | Details |
|:---|:---|
| **University** | SRH University |
| **Student Name** | Divya Koriya|
| **Matriculation Number** | 100002007 |
| **Subject** | Reinforcement Learning |
| **Project** | Project 2 — Autonomous Drone Landing (LunarLander) |
| **Date** | 27 February 2026 |

---

## 📌 Project Overview

This project implements a **Deep Q-Network (DQN)** agent to solve the autonomous drone landing problem using the `LunarLander-v3` Gymnasium environment as a digital twin.

The agent learns to:
- 🛬 Land safely on a charging pad
- ⚡ Minimize fuel (battery) consumption
- 💥 Avoid crashes that cause equipment damage and safety risks

> **Key Principle:** This project focuses on building a *responsible autonomous decision system*, not just maximizing a game score. Strong system design — monitoring, ethics, and deployment — is treated as equally important as model performance.

---

## 🗂️ Repository Structure

```
project-reinforce-learning/
│
├── reinforcement_learning.ipynb   # Main Jupyter notebook (all code + outputs)
├── lunar_lander_dqn.h5            # Saved trained model weights
├── learning_curve.png             # Training reward curve plot
├── monitoring_dashboard.png       # Operational monitoring simulation
└── README.md                      # This file
```

---

## 🌍 Environment: LunarLander-v3

The `LunarLander-v3` environment from [Gymnasium](https://gymnasium.farama.org/) simulates a lander descending onto a landing pad. It acts as a **digital twin** of a real warehouse drone system.

### State Space (8 features)

| Index | Feature | Description |
|:---:|:---|:---|
| 0 | X Position | Horizontal position of the lander |
| 1 | Y Position | Vertical height above ground |
| 2 | X Velocity | Horizontal speed |
| 3 | Y Velocity | Vertical speed (descent rate) |
| 4 | Angle | Tilt/orientation of the lander |
| 5 | Angular Velocity | Rotation speed |
| 6 | Left Leg Contact | 1 if left leg touching ground |
| 7 | Right Leg Contact | 1 if right leg touching ground |

### Action Space (4 discrete actions)

| Action | Description |
|:---:|:---|
| 0 | Do nothing (coast) |
| 1 | Fire left engine (push right) |
| 2 | Fire main engine (push up) |
| 3 | Fire right engine (push left) |

### Reward Structure

| Event | Reward |
|:---|:---|
| Safe landing on pad | +100 to +140 |
| Crash | -100 |
| Each leg contact with ground | +10 |
| Each thruster fire (fuel cost) | -0.3 |
| Episode solved threshold | ≥ 200 average |

---

## 🤖 RL Problem Formulation

| RL Concept | LunarLander-v3 | Real Drone System |
|:---|:---|:---|
| **Agent** | Lander controller | Autonomous warehouse drone |
| **Environment** | Physics simulator | Warehouse + real-world physics |
| **State** | Position, velocity, angle, leg contacts | Drone pose + IMU + LiDAR sensors |
| **Action** | Thruster commands (0–3) | Motor speed commands |
| **Reward** | Safe landing + fuel efficiency | Successful dock + energy saved |
| **Episode** | One complete descent attempt | One landing mission |

---

## 🧠 Algorithm: Deep Q-Network (DQN)

DQN uses a neural network to approximate Q-values — the expected cumulative reward for taking action `a` in state `s`:

```
Q(state, action) ≈ Neural Network(state) → [Q-value for each action]
```

### Key Components

**1. Q-Network Architecture**
```
Input (8)  →  Dense(64, ReLU)  →  Dense(64, ReLU)  →  Output(4, Linear)
```

**2. ε-Greedy Exploration**
- Starts fully exploratory (ε = 1.0)
- Decays over time (ε × 0.995 per episode)
- Floors at minimum exploration (ε = 0.01)

**3. Experience Replay**
- Stores past `(state, action, reward, next_state, done)` transitions
- Random mini-batch sampling breaks temporal correlation
- Buffer size: 10,000 transitions

**4. Target Network**
- Separate copy of Q-network updated every 20 episodes
- Provides stable training targets, prevents divergence

### Hyperparameters

| Parameter | Value | Purpose |
|:---|:---|:---|
| Learning Rate | 0.001 | Adam optimizer step size |
| Discount Factor γ | 0.99 | How much future rewards are valued |
| ε Start | 1.0 | Begin fully exploratory |
| ε Minimum | 0.01 | Always keep small exploration |
| ε Decay | 0.995 | Per-episode decay rate |
| Batch Size | 32 | Mini-batch training samples |
| Replay Buffer | 10,000 | Past experience storage |
| Target Update | Every 20 eps | Stable target network |
| Max Steps/Episode | 500 | Cap for training speed |

---

## 📈 Training Results

Training ran for **150 episodes** on CPU (Python 3.10.10, TensorFlow 2.20.0).

| Metric | Value |
|:---|:---|
| Total Episodes | 150 |
| Best Score | 89.09 |
| Final Average (last 100 eps) | -83.11 |
| Score Improvement | ~59% over training |
| Crashes (score ≤ -100) | 69 / 150 (46%) — mostly early episodes |

### Learning Phases

| Phase | Episodes | Behaviour |
|:---|:---|:---|
| Pure Exploration | 1–50 | Random actions, scores -250 to -380, ε ≈ 1.0 decaying |
| Policy Learning | 51–100 | ε = 0.01, scores improve -148 → +88, first non-crash episodes |
| Steady Improvement | 101–150 | Average rises -204 → -83 (+59%), no late crashes |

> **Note:** 150 episodes demonstrates clear learning behaviour. Scores above 200 (fully solved) would require 500–1000+ episodes. This run was optimised for speed on CPU hardware.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10.10
- pip

### Install Dependencies

```bash
pip install gymnasium[box2d] tensorflow numpy matplotlib
```

### Verify Versions

```python
import tensorflow as tf
import gymnasium as gym
import numpy as np

print(tf.__version__)   # 2.20.0
print(gym.__version__)  # 1.2.3
print(np.__version__)   # 2.2.6
```

### Known Issue

> ⚠️ `LunarLander-v2` is deprecated in Gymnasium ≥ 0.27. Use `LunarLander-v3` instead (identical behaviour, updated version string).

---

## ▶️ How to Run

1. Clone the repository
```bash
git clone <your-repo-url>
cd project-reinforce-learning
```

2. Install dependencies
```bash
pip install gymnasium[box2d] tensorflow numpy matplotlib
```

3. Launch Jupyter
```bash
jupyter notebook reinforcement_learning.ipynb
```

4. Run all cells top to bottom (**Kernel → Restart & Run All**)

> ⏱️ Training takes approximately **2–5 minutes** on a standard CPU with the default 150 episodes.

---

## 💾 Saving & Loading the Model

The trained model is saved automatically at the end of the notebook:

```python
# Save
agent.model.save('lunar_lander_dqn.h5')

# Load later
from tensorflow import keras
loaded_model = keras.models.load_model('lunar_lander_dqn.h5')
```

---

## 🏗️ End-to-End System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 TRAINING PHASE  (Cloud / GPU)                │
│                                                              │
│   LunarLander-v3  →  DQN Agent  →  Saved Policy (.h5)       │
│   (Simulator)        (TensorFlow)   (Model export)          │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ Deploy (TFLite conversion)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                INFERENCE PHASE  (Edge / On-Drone)            │
│                                                              │
│   Drone Sensors  →  Loaded Policy  →  Motor Commands        │
│   (IMU, LiDAR)      (TFLite)          (Thruster 0–3)        │
└──────────────────────────────────────────────────────────────┘

         MONITORING:  Logs → Dashboard → Alert → Retrain
```

| Decision | Choice | Reason |
|:---|:---|:---|
| Training location | Cloud GPU | DQN is compute-heavy |
| Inference location | Edge device (on drone) | Real-time <5ms latency needed |
| Model format | TFLite | Lightweight for embedded hardware |
| Update frequency | Weekly retraining | Balance stability vs. adaptation |
| Fallback | Rule-based PID controller | Safety net if RL policy fails |

---

## 📡 Monitoring & Maintenance

| Metric | Normal Range | Alert Threshold | Action |
|:---|:---|:---|:---|
| Landing success rate | > 85% | < 75% | Schedule retraining |
| Crash rate | < 5% | > 10% | Emergency shutdown |
| Avg fuel usage | < 130 units | > 150 units | Investigate reward hacking |
| Episode duration | < 600 steps | > 900 steps | Check hover behaviour |

**Drift Detection:** Rolling 7-day average monitored vs. baseline. Degradation > 15% triggers the retraining pipeline automatically.

**After Repeated Failures:** Switch to PID fallback → alert human supervisor → collect failure logs → retrain with augmented data.

---

## ⚖️ Ethics & Risk Analysis

### 🦺 Safety
- Human workers share warehouse airspace with drones
- Hardware-level emergency kill switch operates independently of the RL policy
- Geofencing ensures drones stay above worker zones until landing pad is clear
- Mandatory LED and audio warnings active during all descent phases

### ⚠️ Reward Hacking
The agent may learn to **hover indefinitely** to avoid the -100 crash penalty — appearing "safe" while wasting fuel and blocking the pad.

**Mitigations:**
- Per-step time penalty (-0.1/step) to discourage hovering
- Shaped reward for downward progress toward the pad
- Maximum episode cap forces landing attempt

### 🔍 Transparency
- Every action, state, and reward logged with timestamp
- Any crash episode can be replayed from stored logs
- Model version tracked at each deployment — incidents are traceable

### 👤 Responsibility Chain
| Role | Responsibility |
|:---|:---|
| Fleet Manager | Reviews daily monitoring dashboard |
| ML Engineer | Triggers retraining when metrics degrade |
| Safety Officer | Authority to ground all drones immediately |
| System | Never runs without a named human owner assigned |

---

## 📋 Project Checklist

| Deliverable | Status |
|:---|:---|
| Environment setup (LunarLander-v3) | ✅ Complete |
| RL Problem Formulation + Mapping Table | ✅ Complete |
| Business Interpretation | ✅ Complete |
| DQN Implementation (TensorFlow) | ✅ Complete |
| ε-greedy Exploration + Experience Replay | ✅ Complete |
| Learning Curve Plot | ✅ Complete |
| System Architecture Diagram | ✅ Complete |
| Monitoring & Maintenance Section | ✅ Complete |
| Ethics & Risk Analysis | ✅ Complete |
| Model Save / Load | ✅ Complete |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|:---|:---|:---|
| Python | 3.10.10 | Core language |
| TensorFlow / Keras | 2.20.0 | Neural network implementation |
| Gymnasium | 1.2.3 | RL environment |
| NumPy | 2.2.6 | Numerical operations |
| Matplotlib | latest | Plotting & visualisation |

---

## 📚 References

- [Gymnasium LunarLander Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning*, Nature
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd Edition)

---

<div align="center">

*"A simple DQN with strong system design beats a complex model with weak real-world reasoning."*

**SRH University · Divya Koriya· Reinforcement Learning · 27 February 2026**

</div>

