"""
VPG (REINFORCE with Baseline) for OBELIX - Difficulty 3
Features: 4-Frame Stacking + Return Normalization + Trajectory Batching
"""

from __future__ import annotations
import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS     = 18
STACK_SIZE = 4
N_OBS_STACKED = N_OBS * STACK_SIZE  # 72 inputs

# -----------------------------------------------------------------------
# Policy Network 
# -----------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=[128, 64]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def select_action(self, s):
        s_t   = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        probs = self.forward(s_t).squeeze(0)
        dist  = Categorical(probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy()

    def select_greedy_action(self, s):
        s_t   = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
        probs = self.forward(s_t).squeeze(0)
        return int(torch.argmax(probs).item())

# -----------------------------------------------------------------------
# Value Network 
# -----------------------------------------------------------------------
class ValueNetwork(nn.Module):
    def __init__(self, in_dim, hidden=[256, 128]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# -----------------------------------------------------------------------
# VPG Agent
# -----------------------------------------------------------------------
class VPG:
    def __init__(self, OBELIX, args):
        self.OBELIX   = OBELIX
        self.args     = args
        self.gamma    = args.gamma
        self.beta     = args.beta
        self.seed     = args.seed

        self.pNetwork = PolicyNetwork(N_OBS_STACKED, N_ACTIONS, hidden=[128, 128])
        self.vNetwork = ValueNetwork(N_OBS_STACKED, hidden=[128, 128])

        self.policyOptimizer = optim.Adam(self.pNetwork.parameters(), lr=args.policy_lr)
        self.valueOptimizer  = optim.Adam(self.vNetwork.parameters(), lr=args.value_lr)

        self.initBookKeeping()

        # Batching variables
        self.batch_returns = []
        self.batch_logProbs = []
        self.batch_entropies = []
        self.batch_values = []

    def initBookKeeping(self):
        self.ep_reward  = []
        self.ep_length  = []
        self.scores     = []
        self.start_time = time.time()

    def performBookKeeping(self, episode, episode_reward, steps_taken):
        self.ep_reward.append(episode_reward)
        self.ep_length.append(steps_taken)
        
        if (episode + 1) % 50 == 0:
            mean100 = np.mean(self.ep_reward[-100:])
            ok      = sum(1 for r in self.ep_reward[-50:] if r > 500)
            elapsed = time.time() - self.start_time
            speed   = (episode + 1) / max(0.001, elapsed)
            remain  = (self.args.episodes - episode - 1) / max(0.001, speed) / 60
            print(f"Ep {episode+1:5d}/{self.args.episodes} | "
                  f"Avg(100): {mean100:7.1f} | "
                  f"ok(50): {ok:2d}/50 | "
                  f"{speed:4.1f} ep/s | "
                  f"~{remain:4.1f} min left")

    def trainPolicyAndValueNetworks(self):
        if not self.batch_returns: return

        returns   = torch.tensor(self.batch_returns, dtype=torch.float32)
        logProbs  = torch.stack(self.batch_logProbs).squeeze()
        entropies = torch.stack(self.batch_entropies).squeeze()
        values    = torch.stack(self.batch_values).squeeze()

        # --- THE MATH STABILIZER (Return Normalization) ---
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Advantage = returns - baseline
        deltas = returns - values.detach()

        # Policy loss with entropy regularization
        baselineLoss = -1.0 * (deltas * logProbs).mean()
        entropyLoss  = -1.0 * entropies.mean()
        policyLoss   = baselineLoss + self.beta * entropyLoss

        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        nn.utils.clip_grad_norm_(self.pNetwork.parameters(), 1.0)
        self.policyOptimizer.step()

        # Value loss (MSE)
        valueLoss = 0.5 * ((returns - values) ** 2).mean()

        self.valueOptimizer.zero_grad()
        valueLoss.backward()
        self.valueOptimizer.step()

        # Clear batch data
        self.batch_returns, self.batch_logProbs, self.batch_entropies, self.batch_values = [], [], [], []

    def trainAgent(self):
        update_every = 10  # Train the network after every 10 episodes

        for episode in range(self.args.episodes):
            env = self.OBELIX(
                scaling_factor=self.args.scaling_factor,
                max_steps=self.args.max_steps,
                wall_obstacles=self.args.wall_obstacles,
                difficulty=self.args.difficulty,
                seed=self.seed + episode,
            )
            
            # Initialize 4-Frame Stack
            obs_raw   = np.asarray(env.reset(seed=self.seed + episode), dtype=np.float32)
            obs_queue = deque([obs_raw] * STACK_SIZE, maxlen=STACK_SIZE)
            s         = np.concatenate(obs_queue)

            ep_rewards   = []
            ep_logProbs  = []
            ep_entropies = []
            ep_values    = []

            episode_reward = 0.0
            steps_taken    = 0
            done           = False

            while not done and steps_taken < self.args.max_steps:
                a, logp_a, entropy_pa = self.pNetwork.select_action(s)

                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                value    = self.vNetwork(s_tensor).squeeze()

                obs_next_raw, r, done = env.step(ACTIONS[a], render=False)
                
                # Update Stack
                obs_queue.append(np.asarray(obs_next_raw, dtype=np.float32))
                s_next = np.concatenate(obs_queue)

                ep_rewards.append(float(r))
                ep_logProbs.append(logp_a)
                ep_entropies.append(entropy_pa)
                ep_values.append(value)

                episode_reward += float(r)
                steps_taken    += 1
                s               = s_next

            # Calculate Discounted Returns for THIS episode specifically
            returns = []
            G = 0
            for r in reversed(ep_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            # Add to the global batch
            self.batch_returns.extend(returns)
            self.batch_logProbs.extend(ep_logProbs)
            self.batch_entropies.extend(ep_entropies)
            self.batch_values.extend(ep_values)

            # TRAJECTORY BATCHING: Only backpropagate every 10 episodes
            if (episode + 1) % update_every == 0:
                self.trainPolicyAndValueNetworks()

            self.performBookKeeping(episode, episode_reward, steps_taken)

    def evaluateAgent(self):
        print("\nRunning Evaluation...")
        rewards = []
        for e in range(self.args.eval_episodes):
            env = self.OBELIX(
                scaling_factor=self.args.scaling_factor,
                max_steps=self.args.max_steps,
                wall_obstacles=self.args.wall_obstacles,
                difficulty=self.args.difficulty,
                seed=self.seed + 10000 + e,
            )
            obs_raw   = np.asarray(env.reset(seed=self.seed + 10000 + e), dtype=np.float32)
            obs_queue = deque([obs_raw] * STACK_SIZE, maxlen=STACK_SIZE)
            s         = np.concatenate(obs_queue)
            
            rs = 0.0
            done = False
            steps = 0
            while not done and steps < self.args.max_steps:
                a          = self.pNetwork.select_greedy_action(s)
                obs_next_raw, r, done = env.step(ACTIONS[a], render=False)
                
                obs_queue.append(np.asarray(obs_next_raw, dtype=np.float32))
                s          = np.concatenate(obs_queue)
                
                rs += float(r)
                steps += 1
            rewards.append(rs)
        return np.mean(rewards), np.std(rewards)

    def runVPG(self):
        print("Started VPG Training (Difficulty 3)")
        self.trainAgent()
        final_score, final_std = self.evaluateAgent()
        print(f"\nFinal Evaluation Score: {final_score:.1f} ± {final_std:.1f}")
        return final_score, final_std

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   default="./obelix.py")
    ap.add_argument("--out",            type=str,   default="vpg_diff3_weights.pth")
    ap.add_argument("--episodes",       type=int,   default=10000) # Increased to 10k
    ap.add_argument("--eval_episodes",  type=int,   default=10)
    ap.add_argument("--max_steps",      type=int,   default=600)
    ap.add_argument("--difficulty",     type=int,   default=3)
    ap.add_argument("--wall_obstacles", action="store_true", default=True) # Forced True for Diff 3
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--beta",           type=float, default=0.01)
    ap.add_argument("--policy_lr",      type=float, default=1e-3)
    ap.add_argument("--value_lr",       type=float, default=1e-3)
    ap.add_argument("--seed",           type=int,   default=42)
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)

    agent = VPG(OBELIX, args)
    agent.runVPG()

    torch.save(agent.pNetwork.state_dict(), args.out)
    print("Saved Policy Weights:", args.out)

if __name__ == "__main__":
    main()