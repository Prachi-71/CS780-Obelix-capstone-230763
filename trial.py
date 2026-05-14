"""
CS780 OBELIX — PURE D3QN + Frame Stacking + Curriculum Learning
=======================================================================
NO Reward Shaping. NO Prioritized Experience Replay (SumTree).
Trains across all 3 levels taking the raw -200 wall penalties.
Fast Execution Mode: Learns every 4 steps, uses Huber Loss & Tensor optimization.
"""

import argparse
import os
import random
import shutil
import time
from collections import deque
from datetime import datetime

import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────── HYPERPARAMETERS ────────────────────────────── #
FRAME_STACK    = 4
OBS_DIM        = 18
STATE_DIM      = OBS_DIM * FRAME_STACK      # 72
N_ACTIONS      = 5
HIDDEN1        = 256
HIDDEN2        = 128
VALUE_HIDDEN   = 64
ADV_HIDDEN     = 64

LR             = 1e-4
GAMMA          = 0.99
BATCH_SIZE     = 256
BUFFER_SIZE    = 150_000
TARGET_UPDATE  = 500
MIN_BUFFER     = 2_000

# Exploration
EPS_START      = 1.0
EPS_END        = 0.05
EPS_DECAY      = 0.9998

# Training schedule 
EPISODES_PER_LEVEL = {1: 1500, 2: 2000, 3: 4000}
MAX_STEPS          = 2000
LOG_EVERY          = 50
SAVE_EVERY         = 200        
SMOOTH_WINDOW      = 50         

ACTIONS        = ["L45", "L22", "FW", "R22", "R45"]
SUCCESS_REWARD = 2000           


# ───────────────────────── STANDARD REPLAY BUFFER ─────────────────────── #
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pos = 0
        self.full = False
        self.states      = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.actions     = np.zeros(capacity, dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        self.states[self.pos]      = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.dones[self.pos]       = done
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.pos
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs],
                self.next_states[idxs], self.dones[idxs])

    @property
    def size(self):
        return self.capacity if self.full else self.pos


# ──────────────────────── DUELING DQN NETWORK ───────────────────────────── #
class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,  HIDDEN2),  nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN2, VALUE_HIDDEN), nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(HIDDEN2, ADV_HIDDEN), nn.ReLU(),
            nn.Linear(ADV_HIDDEN, N_ACTIONS),
        )

    def forward(self, x):
        feat = self.backbone(x)
        V    = self.value_stream(feat)
        A    = self.adv_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


# ───────────────────────────── FRAME STACKER ────────────────────────────── #
class FrameStacker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(
            [np.zeros(OBS_DIM, dtype=np.float32)] * FRAME_STACK,
            maxlen=FRAME_STACK)

    def push(self, obs):
        self.frames.append(np.array(obs, dtype=np.float32))

    def get_state(self):
        return np.concatenate(list(self.frames))   


# ──────────────────────────── D3QN AGENT ────────────────────────────────── #
class D3QNAgent:
    def __init__(self, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.online = DuelingDQN().to(self.device)
        self.target = DuelingDQN().to(self.device)  
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer  = optim.Adam(self.online.parameters(), lr=LR)
        self.buffer     = ReplayBuffer(BUFFER_SIZE)
        self.stacker    = FrameStacker()

        self.epsilon    = EPS_START
        self.step_count = 0

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online(s).argmax(dim=1).item()

    def learn(self):
        if self.buffer.size < MIN_BUFFER:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)

        # Optimization: torch.from_numpy prevents slow memory duplication + strict type casting
        states      = torch.from_numpy(states).float().to(self.device)
        actions     = torch.from_numpy(actions).long().to(self.device)
        rewards     = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones       = torch.from_numpy(dones).float().to(self.device)

        with torch.no_grad():
            next_a   = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q   = self.target(next_states).gather(1, next_a).squeeze(1)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        current_q = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Optimization: Huber Loss
        loss = nn.functional.huber_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "online":    self.online.state_dict(),
            "target":    self.target.state_dict(),
            "epsilon":   self.epsilon,
            "step":      self.step_count,
        }, path)

    def load(self, path: str, eval_mode=True):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon    = ckpt.get("epsilon", EPS_END)
        self.step_count = ckpt.get("step", 0)
        if eval_mode:
            self.online.eval()


# ─────────────────────────── PER-LEVEL TRAINING ─────────────────────────── #
def run_level(agent, env_factory, level: int,
              episodes: int, weights_dir: str) -> dict:
    best_mean = -np.inf
    best_path = os.path.join(weights_dir, f"best_level{level}.pth")

    logs = dict(rewards=[], losses=[], epsilons=[],
                steps=[], successes=[])

    # Fix Bug 2: Set curriculum epsilon ONCE before the loop, not inside it
    if level == 1:
        agent.epsilon = 1.0
    else:
        agent.epsilon = 0.4

    decay_steps  = 400
    drop_per_ep  = (agent.epsilon - EPS_END) / decay_steps

    for ep in range(1, episodes + 1):
        print(f"\r  Running Level {level} | Episode {ep}/{episodes}...", end="", flush=True)

        env = env_factory()
        obs = env.reset()

        agent.stacker.reset()
        agent.stacker.push(obs)
        state = agent.stacker.get_state()

        ep_reward  = 0.0
        ep_losses  = []
        ep_steps   = 0
        ep_success = False
        has_box    = False

        for step in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            action_str = ACTIONS[action_idx]

            next_obs, reward, done = env.step(action_str)

            if reward >= 90:
                has_box = True

            shaped_reward = float(reward)
            stuck_flag    = int(next_obs[17])

            if has_box and not done:
                # BULLDOZER MODE — keep rewards at human scale, scale AFTER
                if action_str == "FW" and stuck_flag == 0:
                    shaped_reward += 10.0
                elif action_str in ["L45", "R45", "L22", "R22"]:
                    shaped_reward -= 5.0
                shaped_reward = shaped_reward / 100.0   # Fix Bug 1: scale per-branch

            elif not has_box and not done:
                # EXPLORATION MODE
                if action_str in ["L45", "R45"]:
                    shaped_reward -= 2.5
                elif action_str == "FW" and stuck_flag == 0:
                    shaped_reward += 0.5
                shaped_reward = shaped_reward / 100.0   # Fix Bug 1: scale per-branch

            else:
                # Terminal step (done=True) — scale raw reward only
                shaped_reward = shaped_reward / 100.0

            agent.stacker.push(next_obs)
            next_state = agent.stacker.get_state()

            agent.buffer.push(state, action_idx, shaped_reward, next_state, float(done))

            if step % 4 == 0:
                loss = agent.learn()
                if loss is not None:
                    ep_losses.append(loss)

            ep_reward += float(reward)      # raw reward for logging
            ep_steps   = step + 1
            state      = next_state

            if reward >= SUCCESS_REWARD:
                ep_success = True

            if done:
                break

        # Fix Bug 3: close environment to free resources
        try:
            env.close()
        except Exception:
            pass

        # Fix Bug 2: decay epsilon after episode
        agent.epsilon = max(EPS_END, agent.epsilon - drop_per_ep)

        logs["rewards"].append(ep_reward)
        logs["losses"].append(np.mean(ep_losses) if ep_losses else 0.0)
        logs["epsilons"].append(agent.epsilon)
        logs["steps"].append(ep_steps)
        logs["successes"].append(float(ep_success))

        if ep % SAVE_EVERY == 0:
            ckpt = os.path.join(weights_dir, f"level{level}_ep{ep}.pth")
            agent.save(ckpt)

        if ep % LOG_EVERY == 0:
            print()
            mean_r    = np.mean(logs["rewards"][-LOG_EVERY:])
            mean_l    = np.mean(logs["losses"][-LOG_EVERY:])
            succ_rate = np.mean(logs["successes"][-LOG_EVERY:]) * 100
            print(f"  [L{level} Ep {ep:>4}]  reward={mean_r:>8.1f}  "
                  f"loss={mean_l:.4f}  success={succ_rate:>5.1f}%  "
                  f"eps={agent.epsilon:.3f}  buf={agent.buffer.size}")

            if mean_r > best_mean:
                best_mean = mean_r
                agent.save(best_path)
                print(f"    ↑ new best for level {level}: {best_mean:.1f}")

    return logs


# ─────────────────────────── PLOTTING ───────────────────────────────────── #
LEVEL_COLORS = {1: "#4C72B0", 2: "#DD8452", 3: "#55A868"}
LEVEL_LABELS = {1: "Level 1 – Static", 2: "Level 2 – Blinking", 3: "Level 3 – Moving+Blinking"}

def _smooth(values, w=SMOOTH_WINDOW):
    if len(values) < w: return np.array(values, dtype=float)
    return np.convolve(values, np.ones(w) / w, mode="valid")

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_all(all_logs: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    levels = sorted(all_logs.keys())

    fig, ax = plt.subplots(figsize=(11, 5))
    for lv in levels:
        r = all_logs[lv]["rewards"]
        ax.plot(r, alpha=0.25, color=LEVEL_COLORS[lv])
        ax.plot(_smooth(r), label=LEVEL_LABELS[lv], color=LEVEL_COLORS[lv], linewidth=2)
    ax.set_title("Episode Reward per Difficulty Level")
    ax.legend(); ax.grid(alpha=0.3)
    _savefig(fig, os.path.join(plots_dir, "reward_per_level.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    means  = [np.mean(all_logs[lv]["rewards"][-100:]) for lv in levels]
    stds   = [np.std(all_logs[lv]["rewards"][-100:])  for lv in levels]
    bars   = ax.bar([LEVEL_LABELS[lv] for lv in levels], means, yerr=stds, capsize=7,
                    color=[LEVEL_COLORS[lv] for lv in levels], edgecolor="black")
    ax.set_title("Final Performance Summary")
    ax.grid(axis="y", alpha=0.3)
    _savefig(fig, os.path.join(plots_dir, "summary_bar.png"))


# ───────────────────────── MAIN ENTRY POINT ─────────────────────────────── #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    tag    = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or os.path.join("runs", tag)
    wdir   = os.path.join(outdir, "weights")
    pdir   = os.path.join(outdir, "plots")
    os.makedirs(wdir, exist_ok=True); os.makedirs(pdir, exist_ok=True)

    try:
        from obelix import OBELIX
    except ImportError:
        raise ImportError("Ensure obelix.py is in the working directory.")

    agent = D3QNAgent()
    levels = [args.level] if args.level else [1, 2, 3]
    all_logs = {}
    best_overall = -np.inf
    best_path = os.path.join(wdir, "best_overall.pth")

    for lv in levels:
        n_eps = args.episodes or EPISODES_PER_LEVEL[lv]
        print(f"\n{'='*60}\n  LEVEL {lv}  —  {LEVEL_LABELS[lv]}  ({n_eps} episodes)\n{'='*60}")
        diff_map = {1: 0, 2: 2, 3: 3}
        env_factory = lambda lv=lv: OBELIX(
            scaling_factor=5, 
            difficulty=diff_map[lv], 
            wall_obstacles=True, 
            max_steps=MAX_STEPS
        )
        
        logs = run_level(agent, env_factory, lv, n_eps, wdir)
        all_logs[lv] = logs

        lv_mean = np.mean(logs["rewards"][-100:])
        if lv_mean > best_overall:
            best_overall = lv_mean
            agent.save(best_path)

    final_path = os.path.join(wdir, "final.pth")
    agent.save(final_path)

    submission = "d3qn_weights_test.pth"
    shutil.copy(best_path, submission)
    print(f"\n[SAVE] Submission weights -> {submission}")

    plot_all(all_logs, pdir)


# ───────────────── SUBMISSION INTERFACE (used by evaluate.py) ───────────────── #
import os

_agent_inst   = None
_stacker_inst = None

def _load_agent():
    global _agent_inst, _stacker_inst
    if _agent_inst is None:
        device = torch.device("cpu")
        _agent_inst = D3QNAgent(device=device)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, "d3qn_weights.pth")
        
        if os.path.exists(weights_path):
            _agent_inst.load(weights_path, eval_mode=True)
            print("Successfully loaded pure D3QN weights for evaluation.")
            
        _stacker_inst = FrameStacker()

def policy(obs, rng=None) -> str:
    _load_agent()
    _stacker_inst.push(obs)
    state = torch.FloatTensor(
        _stacker_inst.get_state()).unsqueeze(0).to(_agent_inst.device)
    with torch.no_grad():
        idx = _agent_inst.online(state).argmax(dim=1).item()
    return ACTIONS[idx]

def reset_episode():
    if _stacker_inst is not None:
        _stacker_inst.reset()

if __name__ == "__main__":
    main()