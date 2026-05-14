"""
D3QN + PER (Dueling Double Deep Q-Network with Prioritized Experience Replay)
For OBELIX - Difficulty 3
Optimized for Sample Efficiency & GPU Acceleration
"""

import argparse
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS = 18
STACK_SIZE = 4
N_OBS_STACKED = N_OBS * STACK_SIZE

# -----------------------------------------------------------------------
# Dueling Q-Network Architecture
# -----------------------------------------------------------------------
class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNetwork, self).__init__()
        
        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        
        # Value Stream: "How good is this state overall?"
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream: "How much better is this action than the others?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# -----------------------------------------------------------------------
# Prioritized Experience Replay (PER) Buffer
# -----------------------------------------------------------------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], []

        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors, offset=1e-5):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + offset

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------------------
# Environment Loader
# -----------------------------------------------------------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# -----------------------------------------------------------------------
# D3QN Agent
# -----------------------------------------------------------------------
class D3QNAgent:
    def __init__(self, OBELIX, args):
        self.OBELIX = OBELIX
        self.args = args
        
        self.active_net = DuelingQNetwork(N_OBS_STACKED, N_ACTIONS).to(device)
        self.target_net = DuelingQNetwork(N_OBS_STACKED, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.active_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.active_net.parameters(), lr=args.lr)
        self.memory = PrioritizedReplayBuffer(args.buffer_size)
        
        self.epsilon = args.eps_start
        self.beta = args.beta_start
        
    def train_step(self):
        if len(self.memory) < self.args.batch_size:
            return
            
        samples, indices, weights = self.memory.sample(self.args.batch_size, self.beta)
        
        states = torch.FloatTensor(np.array([s[0] for s in samples])).to(device)
        actions = torch.LongTensor(np.array([s[1] for s in samples])).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array([s[2] for s in samples])).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array([s[3] for s in samples])).to(device)
        dones = torch.FloatTensor(np.array([s[4] for s in samples])).unsqueeze(1).to(device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        # Double DQN Logic
        with torch.no_grad():
            # 1. Active net picks the best action
            best_actions = self.active_net(next_states).argmax(dim=1, keepdim=True)
            # 2. Target net evaluates that action
            next_q_values = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + (1 - dones) * self.args.gamma * next_q_values

        current_q = self.active_net(states).gather(1, actions)
        
        # TD Error for Priority Updates
        td_errors = (target_q - current_q).squeeze().detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)
        
        # Huber Loss with Importance Sampling weights
        loss = (torch.nn.functional.smooth_l1_loss(current_q, target_q, reduction='none') * weights).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.active_net.parameters(), 1.0)
        self.optimizer.step()

    def run(self):
        print(f"Starting D3QN on {device.type.upper()} | Target: Diff 3")
        start_time = time.time()
        ep_rewards = []
        
        for episode in range(self.args.episodes):
            env = self.OBELIX(
                scaling_factor=self.args.scaling_factor,
                max_steps=self.args.max_steps,
                wall_obstacles=self.args.wall_obstacles,
                difficulty=self.args.difficulty,
                seed=self.args.seed + episode
            )
            
            obs_raw = np.asarray(env.reset(seed=self.args.seed + episode), dtype=np.float32)
            obs_queue = deque([obs_raw]*STACK_SIZE, maxlen=STACK_SIZE)
            state = np.concatenate(obs_queue)
            
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < self.args.max_steps:
                # Epsilon-Greedy Action Selection
                if random.random() < self.epsilon:
                    action_idx = random.randint(0, N_ACTIONS - 1)
                else:
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                        q_vals = self.active_net(state_tensor)
                        action_idx = q_vals.argmax().item()
                        
                next_obs_raw, reward, done = env.step(ACTIONS[action_idx], render=False)
                
                obs_queue.append(np.asarray(next_obs_raw, dtype=np.float32))
                next_state = np.concatenate(obs_queue)
                
                self.memory.add(state, action_idx, reward, next_state, done)
                self.train_step()
                
                state = next_state
                episode_reward += float(reward)
                step += 1
                
            # Update target network
            if episode % self.args.target_update == 0:
                self.target_net.load_state_dict(self.active_net.state_dict())
                
            # Decay Epsilon & Beta
            self.epsilon = max(self.args.eps_end, self.epsilon * self.args.eps_decay)
            self.beta = min(1.0, self.beta + (1.0 - self.args.beta_start) / self.args.episodes)
            
            ep_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_10 = np.mean(ep_rewards[-10:])
                elapsed = time.time() - start_time
                ep_sec = (episode + 1) / elapsed
                rem_min = (self.args.episodes - episode - 1) / ep_sec / 60
                print(f"Ep {episode+1:4d}/{self.args.episodes} | Avg(10): {avg_10:7.1f} | "
                      f"Eps: {self.epsilon:.2f} | {ep_sec:.1f} ep/s | ~{rem_min:.1f}m left")

        return self.active_net

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="d3qn_diff3_weights.pth")
    ap.add_argument("--episodes", type=int, default=1500) # Only 1500 needed now
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true", default=True)
    ap.add_argument("--scaling_factor", type=int, default=5)
    
    # D3QN Hyperparameters
    ap.add_argument("--lr", type=float, default=0.00025)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--buffer_size", type=int, default=50000)
    ap.add_argument("--target_update", type=int, default=10)
    
    # Decay params
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay", type=float, default=0.995)
    ap.add_argument("--beta_start", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent = D3QNAgent(OBELIX, args)
    trained_net = agent.run()

    torch.save(trained_net.state_dict(), args.out)
    print(f"\nSaved Dueling weights to: {args.out}")

if __name__ == "__main__":
    main()