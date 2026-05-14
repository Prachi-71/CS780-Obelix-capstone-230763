"""
D3QN-PER Hyperparameter Tuner for OBELIX
Based on grid search style from CS780 assignment.
Runs multiple configs, saves best weights.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import sys
import os

try:
    from obelix import OBELIX
except ImportError:
    print("ERROR: Run from CS780-OBELIX repo root.")
    sys.exit(1)

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS     = 18
GAMMA     = 0.99
ALPHA_PER = 0.6
BETA_PER  = 0.4
BETA_RATE = 0.001
TAU       = 0.005
EPISODES_PER_TRIAL = 50
MAX_STEPS = 1000

# -----------------------------------------------------------------------
# Grid search configs (same style as assignment)
# -----------------------------------------------------------------------
lrs          = [0.001, 0.0005]
batch_sizes  = [64, 128]
update_freqs = [10, 50]
eps_decays   = [0.99, 0.995]

configs = []
for lr in lrs:
    for bs in batch_sizes:
        for uf in update_freqs:
            for ed in eps_decays:
                configs.append({
                    'lr': lr, 'batch': bs,
                    'update_freq': uf, 'eps_decay': ed
                })

print(f"Total configs: {len(configs)}")
print(f"Episodes per trial: {EPISODES_PER_TRIAL}")
print(f"Estimated time: {len(configs) * EPISODES_PER_TRIAL / 3.9 / 60:.1f} hours")

# -----------------------------------------------------------------------
# Dueling DQN Network
# -----------------------------------------------------------------------
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

# -----------------------------------------------------------------------
# Prioritized Replay Buffer
# -----------------------------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity   = capacity
        self.alpha      = alpha
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0

    def store(self, exp):
        max_p = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        n     = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(n, batch_size, p=probs)
        exps  = [self.buffer[i] for i in idxs]
        w     = (n * probs[idxs]) ** (-beta)
        w    /= w.max()
        return idxs, torch.FloatTensor(w), exps

    def update(self, idxs, priorities):
        for i, p in zip(idxs, priorities):
            self.priorities[i] = p + 1e-6

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------------------
# Run one trial
# -----------------------------------------------------------------------
def run_trial(cfg):
    online = DuelingDQN(N_OBS, N_ACTIONS)
    target = DuelingDQN(N_OBS, N_ACTIONS)
    target.load_state_dict(online.state_dict())

    optimizer = optim.Adam(online.parameters(), lr=cfg['lr'])
    buffer    = PrioritizedReplayBuffer(10000, ALPHA_PER)
    epsilon   = 1.0
    beta      = BETA_PER
    env       = OBELIX(scaling_factor=1, difficulty=0,
                       wall_obstacles=False, max_steps=MAX_STEPS)
    rewards   = []
    successes = 0

    for ep in range(EPISODES_PER_TRIAL):
        obs  = np.asarray(env.reset(), dtype=np.float32)
        done = False
        ep_r = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            step += 1
            if random.random() < epsilon:
                a = random.randint(0, N_ACTIONS - 1)
            else:
                with torch.no_grad():
                    a = int(online(torch.FloatTensor(obs).unsqueeze(0)).argmax().item())

            obs_next, r, done = env.step(ACTIONS[a], render=False)
            obs_next = np.asarray(obs_next, dtype=np.float32)
            ep_r    += r
            buffer.store((obs, a, r, obs_next, float(done)))

            if len(buffer) >= cfg['batch']:
                idxs, weights, exps = buffer.sample(cfg['batch'], beta)
                states      = torch.FloatTensor([e[0] for e in exps])
                actions     = torch.LongTensor([e[1] for e in exps]).unsqueeze(1)
                rew         = torch.FloatTensor([e[2] for e in exps]).unsqueeze(1)
                next_states = torch.FloatTensor([e[3] for e in exps])
                dones       = torch.FloatTensor([e[4] for e in exps]).unsqueeze(1)

                with torch.no_grad():
                    next_a   = online(next_states).argmax(1, keepdim=True)
                    next_q   = target(next_states).gather(1, next_a)
                    targets  = rew + GAMMA * next_q * (1 - dones)

                predicted = online(states).gather(1, actions)
                td_errors = (targets - predicted).abs().detach().squeeze(1).numpy()
                loss = (weights.unsqueeze(1) * nn.HuberLoss(reduction='none')(predicted, targets)).mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10)
                optimizer.step()
                buffer.update(idxs, td_errors)

            obs = obs_next

        epsilon = max(0.05, epsilon * cfg['eps_decay'])
        beta    = min(1.0, beta + BETA_RATE)

        if (ep + 1) % cfg['update_freq'] == 0:
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

        rewards.append(ep_r)
        if ep_r > 500:
            successes += 1

    final_avg = np.mean(rewards[-20:])
    return final_avg, successes, online

# -----------------------------------------------------------------------
# Main grid search
# -----------------------------------------------------------------------
if __name__ == "__main__":
    best_reward = -float('inf')
    best_cfg    = None
    results     = []
    t_start     = time.time()

    for idx, cfg in enumerate(configs):
        print(f"\n--- Config {idx+1}/{len(configs)} ---")
        print(f"  lr={cfg['lr']} batch={cfg['batch']} "
              f"update_freq={cfg['update_freq']} eps_decay={cfg['eps_decay']}")

        t0          = time.time()
        avg, ok, net = run_trial(cfg)
        elapsed     = (time.time() - t0) / 60

        print(f"  AvgR(last20)={avg:.1f} | ok={ok}/{EPISODES_PER_TRIAL} | {elapsed:.1f} min")
        results.append({'cfg': cfg, 'avg': avg, 'ok': ok})

        if avg > best_reward:
            best_reward = avg
            best_cfg    = cfg
            torch.save(net.state_dict(), "d3qn_weights.pth")
            print(f"  --> NEW BEST! Saved d3qn_weights.pth")

    # Sort and print results
    results.sort(key=lambda x: x['avg'], reverse=True)
    print(f"\n{'='*60}")
    print(f"  TUNING COMPLETE | Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")
    print(f"\nAll configs ranked:")
    for i, r in enumerate(results):
        print(f"  {i+1}. AvgR={r['avg']:8.1f} ok={r['ok']:3d} | {r['cfg']}")

    print(f"\nBest config: {best_cfg}")
    print(f"Best AvgR:   {best_reward:.1f}")
    print(f"\nNow run train_d3qn.py with these hyperparameters for full training.")
