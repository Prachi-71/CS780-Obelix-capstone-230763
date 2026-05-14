"""
AUTOMATED HYPERPARAMETER TUNER FOR D3QN (Difficulty 2)
Uses Random Search + Early Stopping. Pure D3QN (No PER).
"""

import numpy as np
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys

try:
    from obelix import OBELIX
except ImportError:
    print("ERROR: Run from CS780-OBELIX repo root.")
    sys.exit(1)

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_total = time.time()

# --- CORE ARCHITECTURE (Dueling Stream Only) ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

# --- STANDARD UNIFORM REPLAY BUFFER (No PER) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        # Uniformly sample random memories without priority bias
        exps = random.sample(self.buffer, batch_size)
        
        s = np.stack([e[0] for e in exps]).astype(np.float32)
        a = np.array([e[1] for e in exps], dtype=np.int64)
        r = np.array([e[2] for e in exps], dtype=np.float32)
        s2 = np.stack([e[3] for e in exps]).astype(np.float32)
        d = np.array([e[4] for e in exps], dtype=np.float32)
        
        return (torch.FloatTensor(s).to(device), 
                torch.LongTensor(a).to(device), 
                torch.FloatTensor(r).to(device), 
                torch.FloatTensor(s2).to(device), 
                torch.FloatTensor(d).to(device))

    def __len__(self): return len(self.buffer)

# --- THE TUNING ENGINE ---
def evaluate_params(lr, tau, batch_size, eps_decay):
    print(f"\nTesting: LR={lr:.5f} | TAU={tau:.4f} | BATCH={batch_size} | DECAY={eps_decay}")
    
    online = DuelingDQN(72, 5).to(device) # 72 = 18 * 4 frames
    target = DuelingDQN(72, 5).to(device)
    target.load_state_dict(online.state_dict())
    
    optimizer = optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(50000)
    
    env = OBELIX(scaling_factor=5, max_steps=300, difficulty=2, wall_obstacles=False)
    rewards = []
    steps = 0
    
    for ep in range(500): # Short runs to test viability
        obs_raw = np.asarray(env.reset(), dtype=np.float32)
        obs_queue = deque([obs_raw] * 4, maxlen=4)
        obs = np.concatenate(obs_queue)
        
        ep_r = 0.0
        done = False
        step = 0
        
        while not done and step < 300:
            step += 1
            steps += 1
            eps = max(0.05, 1.0 - (steps / eps_decay))
            
            if random.random() < eps:
                a = random.randint(0, 4)
            else:
                with torch.no_grad():
                    q = online(torch.FloatTensor(obs).unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
                a = int(np.argmax(q))
                
            obs_next_raw, r, done = env.step(ACTIONS[a], render=False)
            obs_queue.append(np.asarray(obs_next_raw, dtype=np.float32))
            obs_next = np.concatenate(obs_queue)
            
            ep_r += r
            buffer.store((obs, a, r, obs_next, float(done)))
            obs = obs_next
            
            if len(buffer) > batch_size:
                # Standard Sample (No weights, no indexes)
                s_t, a_t, r_t, s2_t, d_t = buffer.sample(batch_size)
                
                with torch.no_grad():
                    next_a = online(s2_t).argmax(1)
                    next_q = target(s2_t).gather(1, next_a.unsqueeze(1)).squeeze(1)
                    targets = r_t + 0.99 * (1.0 - d_t) * next_q
                    
                predicted = online(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                
                # Standard Smooth L1 Loss (No Importance Sampling Weights)
                loss = nn.functional.smooth_l1_loss(predicted, targets)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                
                for target_param, local_param in zip(target.parameters(), online.parameters()):
                    target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
                    
            if done: break
            
        rewards.append(ep_r)
        
        # EARLY STOPPING
        '''''if ep == 50:
            avg_50 = np.mean(rewards[-50:])
            if avg_50 < -100:
                print(f"  [ABORT] Model failing early (Avg: {avg_50:.1f}). Skipping.")
                return -9999, None'''''
       
        if (ep + 1) % 50 == 0:
            avg_50 = np.mean(rewards[-50:])
            elapsed = time.time() - t_total
            
            # Prevent divide-by-zero if your CPU blazes through it
            speed = (ep + 1) / max(0.001, elapsed) 
            
            # Calculate remaining minutes based on 500 total episodes
            remain_min = (500 - (ep + 1)) / max(0.001, speed) / 60 
            
            print(f"Ep {ep+1:3d}/500 | Avg Reward: {avg_50:7.1f} | Speed: {speed:4.1f} ep/s | Time Left: ~{remain_min:4.1f} min")
                
        final_score = np.mean(rewards[-500:])
    print(f"  [SUCCESS] Finished 150 eps. Final Score: {final_score:.1f}")
    return final_score, online.state_dict()

def run_tuner():
    best_score = -float('inf')
    best_params = {}
    
    
    lr = 0.00047
    tau = 0.003
    batch = 256
    decay = 150000
        
    score, weights = evaluate_params(lr, tau, batch, decay)
        
    if score > best_score:
        best_score = score
        best_params = {'LR': lr, 'TAU': tau, 'BATCH': batch, 'DECAY': decay}
        if weights is not None:
                torch.save(weights, "best_d3qn_tuned.pth")
                print("  --> New Best Weights Saved!")

    print(f"\nTUNING COMPLETE. Best Params: {best_params} | Score: {best_score:.1f}")

if __name__ == "__main__":
    run_tuner()