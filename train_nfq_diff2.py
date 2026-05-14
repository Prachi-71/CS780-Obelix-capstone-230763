"""
Neural Fitted Q-Iteration (NFQ) for OBELIX - Difficulty 2
Features: Offline Batch Training + 4-Frame Stacking

Why NFQ for difficulty 3 (blinking box):
  - Offline batch training handles sparse rewards better
  - 4-frame stacking gives agent memory of where box was last seen
  - When box blinks out, stacked frames still carry last known position
  - NFQ collects data first, then trains — more stable than online DQN

Run:
  python train_nfq.py
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
LR                = 0.001
NFQ_ITERATIONS    = 50
EPISODES_PER_ITER = 10
EPOCHS_PER_ITER   = 5
MAX_STEPS         = 600
SCALING_FACTOR    = 5       

EPS_START         = 1.0
EPS_END           = 0.05
EPS_DECAY_ITERS   = 30

ACTIONS           = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS         = 5
N_OBS             = 18
STACK_SIZE        = 4      # frame stacking: agent remembers last 4 obs
N_OBS_STACKED     = N_OBS * STACK_SIZE  # 72

device = torch.device("cpu")

# -----------------------------------------------------------------------
# Q-Network (vanilla NFQ — single stream, no dueling)
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
# NFQ Dataset — grows over all iterations
# -----------------------------------------------------------------------
class NFQDataset:
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_all_tensors(self):
        s  = np.stack([x[0] for x in self.buffer])
        a  = np.array([x[1] for x in self.buffer], dtype=np.int64)
        r  = np.array([x[2] for x in self.buffer], dtype=np.float32)
        s2 = np.stack([x[3] for x in self.buffer])
        d  = np.array([x[4] for x in self.buffer], dtype=np.float32)
        return (
            torch.FloatTensor(s).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(s2).to(device),
            torch.FloatTensor(d).to(device)
        )

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
        difficulty=2,
        wall_obstacles=False
    )

    t_total           = time.time()
    total_episodes    = 0
    all_iter_rewards  = []
    all_iter_success  = []

    print(f"\n{'='*60}")
    print(f"  NFQ | Difficulty 2 | {NFQ_ITERATIONS} Iterations")
    print(f"  Frame Stacking={STACK_SIZE} | SCALING={SCALING_FACTOR}")
    print(f"  Episodes/Iter={EPISODES_PER_ITER} | Epochs/Iter={EPOCHS_PER_ITER}")
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

        # --- PHASE 2: FITTED Q-ITERATION (OFFLINE TRAINING) ---
        if len(dataset.buffer) > 1000:
            s_t, a_t, r_t, s2_t, d_t = dataset.get_all_tensors()

            for epoch in range(EPOCHS_PER_ITER):
                with torch.no_grad():
                    next_q_vals = q_net(s2_t).max(1)[0]
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

    torch.save(q_net.state_dict(), "nfq_weights.pth")
    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: nfq_weights.pth")
    return all_iter_rewards, all_iter_success


if __name__ == "__main__":
    rewards, successes = train_nfq()

    iters = np.arange(1, len(rewards) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(iters, rewards, color='steelblue', linewidth=2, marker='o', markersize=4)
    ax1.set(xlabel='Iteration', ylabel='Avg Reward',
            title='NFQ (Difficulty 2) — Avg Reward per Iteration')
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, successes, color='green', linewidth=2, marker='o', markersize=4)
    ax2.set(xlabel='Iteration', ylabel=f'Successes / {EPISODES_PER_ITER} eps',
            title='NFQ — Successes per Iteration')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nfq_plots.png', dpi=150)
    plt.show()
