import numpy as np
import sys
import time
import random
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
# Hyperparameters — matched to starter code insights
# -----------------------------------------------------------------------
GAMMA          = 0.99
LR             = 0.001
BATCH_SIZE     = 256        # larger batch = more stable
BUFFER_SIZE    = 100_000    # large buffer = diverse experiences
#WARMUP_STEPS   = 2000       # collect before training
TARGET_SYNC    = 2000       # hard update every N steps
EPS_START      = 1.0
EPS_END        = 0.05
#EPS_DECAY_STEPS= 200_000    # step-based decay like starter
NO_EPISODES    = 500
#MAX_STEPS      = 1000
SCALING_FACTOR = 5          # KEY FIX: 28x more sensor coverageNO_EPISODES    = 500
MAX_STEPS      = 600
WARMUP_STEPS   = 500
EPS_DECAY_STEPS= 50_000
 
ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS     = 18
device    = torch.device("cpu")
 
# -----------------------------------------------------------------------
# Dueling DQN Network
# -----------------------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
 
    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)
 
ALPHA_PER = 0.6
BETA_PER  = 0.4
BETA_RATE = 0.001
 
# -----------------------------------------------------------------------
# Prioritized Replay Buffer
# -----------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity   = capacity
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0
        self.beta       = BETA_PER
 
    def store(self, exp):
        max_p = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity
 
    def sample(self, batch_size):
        n     = len(self.buffer)
        probs = self.priorities[:n] ** ALPHA_PER
        probs /= probs.sum()
        idxs  = np.random.choice(n, batch_size, p=probs)
        exps  = [self.buffer[i] for i in idxs]
        w     = (n * probs[idxs]) ** (-self.beta)
        w    /= w.max()
        s  = np.stack([e[0] for e in exps]).astype(np.float32)
        a  = np.array([e[1] for e in exps], dtype=np.int64)
        r  = np.array([e[2] for e in exps], dtype=np.float32)
        s2 = np.stack([e[3] for e in exps]).astype(np.float32)
        d  = np.array([e[4] for e in exps], dtype=np.float32)
        self.beta = min(1.0, self.beta + BETA_RATE)
        return idxs, torch.FloatTensor(w), s, a, r, s2, d
 
    def update_priorities(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.priorities[i] = p + 1e-6
 
    def __len__(self):
        return len(self.buffer)
 
# -----------------------------------------------------------------------
# Epsilon schedule — step based like starter
# -----------------------------------------------------------------------
def eps_by_step(t):
    if t >= EPS_DECAY_STEPS:
        return EPS_END
    frac = t / EPS_DECAY_STEPS
    return EPS_START + frac * (EPS_END - EPS_START)
 
# -----------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------
def train():
    online = DuelingDQN(N_OBS, N_ACTIONS).to(device)
    target = DuelingDQN(N_OBS, N_ACTIONS).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
 
    optimizer = optim.Adam(online.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)
 
    rewards   = []
    successes = 0
    steps     = 0
    t_total   = time.time()
 
    print(f"\n{'='*60}")
    print(f"  D3QN Improved | {NO_EPISODES} eps | scaling={SCALING_FACTOR}")
    print(f"  LR={LR} | BATCH={BATCH_SIZE} | BUFFER={BUFFER_SIZE}")
    print(f"  WARMUP={WARMUP_STEPS} | TARGET_SYNC={TARGET_SYNC}")
    print(f"{'='*60}")
 
    for ep in range(NO_EPISODES):
        # New env per episode with different seed (like starter)
        env = OBELIX(
            scaling_factor=SCALING_FACTOR,
            max_steps=MAX_STEPS,
            difficulty=0,
            wall_obstacles=False,
            seed=ep
        )
        obs  = np.asarray(env.reset(seed=ep), dtype=np.float32)
        done = False
        ep_r = 0.0
        step = 0
 
        while not done and step < MAX_STEPS:
            step  += 1
            steps += 1
            eps    = eps_by_step(steps)
 
            # Epsilon-greedy
            if np.random.rand() < eps:
                a = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    q = online(torch.tensor(obs).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(q))
 
            obs_next, r, done = env.step(ACTIONS[a], render=False)
            obs_next = np.asarray(obs_next, dtype=np.float32)
            ep_r    += r
 
            buffer.store((obs, a, r, obs_next, float(done)))
            obs = obs_next
 
            # Train only after warmup
            if len(buffer) >= max(WARMUP_STEPS, BATCH_SIZE):
                idxs, weights, s_b, a_b, r_b, s2_b, d_b = buffer.sample(BATCH_SIZE)
                s_t  = torch.tensor(s_b)
                a_t  = torch.tensor(a_b)
                r_t  = torch.tensor(r_b)
                s2_t = torch.tensor(s2_b)
                d_t  = torch.tensor(d_b)
 
                # Double DQN target
                with torch.no_grad():
                    next_a   = online(s2_t).argmax(1)
                    next_q   = target(s2_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    targets  = r_t + GAMMA * (1.0 - d_t) * next_q
 
                predicted = online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                td_errors = (targets - predicted).abs().detach().numpy()
 
                # PER weighted loss
                loss = (weights * nn.functional.smooth_l1_loss(predicted, targets, reduction='none')).mean()
 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
 
                buffer.update_priorities(idxs, td_errors)
 
                # Hard target sync every TARGET_SYNC steps
                if steps % TARGET_SYNC == 0:
                    target.load_state_dict(online.state_dict())
 
            if done:
                break
 
        rewards.append(ep_r)
        if ep_r > 500:
            successes += 1
 
        if (ep + 1) % 50 == 0 or (ep + 1) == NO_EPISODES:
            elapsed = time.time() - t_total
            speed   = (ep + 1) / max(0.001, elapsed)
            remain  = (NO_EPISODES - ep - 1) / max(0.001, speed) / 60
            print(f"  ep {ep+1:4d}/{NO_EPISODES} | "
                  f"eps={eps_by_step(steps):.3f} | "
                  f"R={ep_r:8.1f} | "
                  f"AvgR={np.mean(rewards[-50:]):8.1f} | "
                  f"ok={successes:3d} | "
                  f"buf={len(buffer):6d} | "
                  f"{speed:.2f} ep/s | "
                  f"~{remain:.1f} min left",
                  flush=True)
 
    torch.save(online.state_dict(), "d3qn_weights.pth")
    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: d3qn_weights.pth")
    return rewards
 
 
if __name__ == "__main__":
    rewards = train()
 
    w    = 50
    kern = np.ones(w) / w
    rewards = [float(r) for r in rewards if r is not None]
    avg  = np.convolve(rewards, kern, mode='valid')
    succ = np.convolve([1 if r > 500 else 0 for r in rewards], kern, mode='valid')
    eps  = np.arange(1, len(rewards) + 1)
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(eps, rewards, alpha=0.3, color='steelblue', label='Reward')
    ax1.plot(eps[w-1:], avg, color='navy', linewidth=2, label='Avg(50)')
    ax1.set(xlabel='Episode', ylabel='Reward', title='D3QN Improved Reward vs Episodes')
    ax1.legend(); ax1.grid(True, alpha=0.3)
 
    ax2.plot(eps[w-1:], succ * 100, color='green', linewidth=2)
    ax2.set(xlabel='Episode', ylabel='Success %',
            title='Success Rate vs Episodes', ylim=(0, 100))
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('d3qn_improved_plots.png', dpi=150)
    plt.show()