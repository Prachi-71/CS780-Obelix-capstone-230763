"""
Neural Fitted Q-Iteration (NFQ) for OBELIX - Difficulty 3
Features: Offline Mini-Batch Training + 4-Frame Stacking + Frozen Target Network
"""

import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

try:
    from obelix import OBELIX
except ImportError:
    print("ERROR: Run from CS780-OBELIX repo root.")
    sys.exit(1)

# -----------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------
GAMMA             = 0.99
LR                = 0.0003     # Lowered for stable mini-batching
NFQ_ITERATIONS    = 50
EPISODES_PER_ITER = 10
EPOCHS_PER_ITER   = 40       # Increased for mini-batching
MAX_STEPS         = 600
SCALING_FACTOR    = 5        

EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_ITERS   = 30

ACTIONS           = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS         = 5
N_OBS             = 18       # OBELIX ALWAYS outputs 18
STACK_SIZE        = 4        # 4 frames matches agent.py
N_OBS_STACKED     = N_OBS * STACK_SIZE  # Exactly 72 inputs

device = torch.device("cpu")

# -----------------------------------------------------------------------
# Q-Network 
# -----------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------
# NFQ Dataset
# -----------------------------------------------------------------------
class NFQDataset:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

# -----------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------
def train_nfq():
    q_net     = QNetwork(N_OBS_STACKED, N_ACTIONS).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    dataset   = NFQDataset()

    env = OBELIX(
        scaling_factor=SCALING_FACTOR,
        max_steps=MAX_STEPS,
        difficulty=3,
        wall_obstacles=True  # Walls ON for Difficulty 3
    )

    t_total           = time.time()
    total_episodes    = 0
    all_iter_rewards  = []
    all_iter_success  = []

    print(f"\n{'='*60}")
    print(f"  NFQ | Difficulty 3 | {NFQ_ITERATIONS} Iterations")
    print(f"  Frame Stacking={STACK_SIZE} | Inputs={N_OBS_STACKED}")
    print(f"{'='*60}")

    for iteration in range(NFQ_ITERATIONS):
        eps = max(EPS_END,
                  EPS_START - (iteration / EPS_DECAY_ITERS) * (EPS_START - EPS_END))

        iter_rewards  = []
        iter_successes = 0

        # --- PHASE 1: DATA COLLECTION ---
        for ep in range(EPISODES_PER_ITER):
            total_episodes += 1

            obs_raw   = np.asarray(env.reset(seed=total_episodes), dtype=np.float32)
            obs_queue = deque([obs_raw] * STACK_SIZE, maxlen=STACK_SIZE)
            obs       = np.concatenate(obs_queue)

            done  = False
            ep_r  = 0.0
            step  = 0

            while not done and step < MAX_STEPS:
                step += 1
                if np.random.rand() < eps:
                    a = np.random.randint(N_ACTIONS)
                else:
                    with torch.no_grad():
                        q_vals = q_net(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    a = int(torch.argmax(q_vals).item())

                obs_next_raw, r, done = env.step(ACTIONS[a], render=False)
                obs_queue.append(np.asarray(obs_next_raw, dtype=np.float32))
                obs_next = np.concatenate(obs_queue)

                dataset.add(obs, a, r, obs_next, float(done))
                obs   = obs_next
                ep_r += r

            iter_rewards.append(ep_r)
            if ep_r > 500:
                iter_successes += 1

        # --- PHASE 2: FITTED Q-ITERATION (MINI-BATCHES) ---
        if len(dataset.buffer) > 1000:
            
            # Freeze the target network so it doesn't chase its own tail
            target_net = QNetwork(N_OBS_STACKED, N_ACTIONS).to(device)
            target_net.load_state_dict(q_net.state_dict())
            
            for epoch in range(EPOCHS_PER_ITER):
                batch_size = 256
                # Randomly sample a mini-batch
                idxs = np.random.choice(len(dataset.buffer), batch_size, replace=False)
                
                s = np.stack([dataset.buffer[i][0] for i in idxs])
                a = np.array([dataset.buffer[i][1] for i in idxs], dtype=np.int64)
                r = np.array([dataset.buffer[i][2] for i in idxs], dtype=np.float32)
                s2 = np.stack([dataset.buffer[i][3] for i in idxs])
                d = np.array([dataset.buffer[i][4] for i in idxs], dtype=np.float32)

                s_t = torch.FloatTensor(s).to(device)
                a_t = torch.LongTensor(a).to(device)
                r_t = torch.FloatTensor(r).to(device)
                s2_t = torch.FloatTensor(s2).to(device)
                d_t = torch.FloatTensor(d).to(device)

                with torch.no_grad():
                    next_q_vals = target_net(s2_t).max(1)[0]
                    targets     = r_t + GAMMA * (1.0 - d_t) * next_q_vals

                predictions = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                loss        = nn.functional.mse_loss(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

        avg_r = np.mean(iter_rewards)
        all_iter_rewards.append(avg_r)
        all_iter_success.append(iter_successes)

        elapsed = time.time() - t_total
        print(f"  Iter {iteration+1:2d}/{NFQ_ITERATIONS} | "
              f"eps={eps:.2f} | "
              f"data={len(dataset.buffer):6d} | "
              f"AvgR={avg_r:8.1f} | "
              f"ok={iter_successes}/{EPISODES_PER_ITER} | "
              f"{elapsed/60:.1f} min",
              flush=True)

    torch.save(q_net.state_dict(), "nfq_weights_diff3.pth")
    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: nfq_weights_diff3.pth")
    return all_iter_rewards, all_iter_success


if __name__ == "__main__":
    rewards, successes = train_nfq()

    iters = np.arange(1, len(rewards) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(iters, rewards, color='steelblue', linewidth=2, marker='o', markersize=4)
    ax1.set(xlabel='Iteration', ylabel='Avg Reward',
            title='NFQ (Difficulty 3) — Avg Reward per Iteration')
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, successes, color='green', linewidth=2, marker='o', markersize=4)
    ax2.set(xlabel='Iteration', ylabel=f'Successes / {EPISODES_PER_ITER} eps',
            title='NFQ — Successes per Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nfq_plots_diff3.png', dpi=150)
    #plt.show() # Commented out so it doesn't freeze the script at the end