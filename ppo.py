"""
PPO for OBELIX - Difficulty 3 (Moving + Blinking Box)
CS780 Capstone Project

Optimized for 90-minute training on CPU laptop.
Tuned hyperparameters for difficulty 3.

Run:
  python train_ppo.py --obelix_py ./obelix.py --out ppo_weights.pth
"""

import argparse
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cpu")  # CPU only — Codabench is CPU

ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS     = 5
N_OBS         = 18
STACK_SIZE    = 4
N_OBS_STACKED = N_OBS * STACK_SIZE  # 72

# -----------------------------------------------------------------------
# Rollout Buffer
# -----------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states    = []
        self.actions   = []
        self.logprobs  = []
        self.rewards   = []
        self.dones     = []

    def clear(self):
        self.states    = []
        self.actions   = []
        self.logprobs  = []
        self.rewards   = []
        self.dones     = []

    def __len__(self):
        return len(self.rewards)

# -----------------------------------------------------------------------
# Actor-Critic Network
# -----------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        probs  = self.actor(state)
        dist   = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        probs         = self.actor(state)
        dist          = Categorical(probs)
        logprobs      = dist.log_prob(action)
        entropy       = dist.entropy()
        state_values  = self.critic(state)
        return logprobs, state_values, entropy

# -----------------------------------------------------------------------
# Import obelix — same as starter code
# -----------------------------------------------------------------------
def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

# -----------------------------------------------------------------------
# PPO Agent
# -----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, args):
        self.args       = args
        self.policy     = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)
        self.policy_old = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(),  'lr': args.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': args.lr_critic}
        ])

        self.buffer  = RolloutBuffer()
        self.mse     = nn.MSELoss()

    def update(self):
        # Compute discounted returns
        returns = []
        G = 0
        for r, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                G = 0
            G = r + self.args.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        old_states   = torch.FloatTensor(np.array(self.buffer.states)).to(device)
        old_actions  = torch.LongTensor(np.array(self.buffer.actions)).to(device)
        old_logprobs = torch.FloatTensor(np.array(self.buffer.logprobs)).to(device)

        for _ in range(self.args.K_epochs):
            logprobs, values, entropy = self.policy.evaluate(old_states, old_actions)
            values     = values.squeeze()
            ratios     = torch.exp(logprobs - old_logprobs)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.eps_clip,
                                        1 + self.args.eps_clip) * advantages

            loss = (-torch.min(surr1, surr2)
                    + 0.5 * self.mse(values, returns)
                    - self.args.entropy_coef * entropy).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
def train(OBELIX, args):
    agent      = PPOAgent(args)
    ep_rewards = []
    successes  = 0
    time_step  = 0
    t_total    = time.time()

    print(f"\n{'='*60}")
    print(f"  PPO | Difficulty {args.difficulty} | {args.episodes} eps")
    print(f"  lr_actor={args.lr_actor} | lr_critic={args.lr_critic}")
    print(f"  K={args.K_epochs} | clip={args.eps_clip} | update={args.update_timestep}")
    print(f"  scaling={args.scaling_factor} | stack={STACK_SIZE}")
    print(f"{'='*60}")

    for ep in range(args.episodes):
        env = OBELIX(
            scaling_factor=args.scaling_factor,
            max_steps=args.max_steps,
            wall_obstacles=True,          # no wall — faster + simpler
            difficulty=args.difficulty,
            seed=args.seed + ep,
        )

        obs_raw   = np.asarray(env.reset(seed=args.seed + ep), dtype=np.float32)
        obs_queue = deque([obs_raw] * STACK_SIZE, maxlen=STACK_SIZE)
        state     = np.concatenate(obs_queue)

        ep_r  = 0.0
        done  = False
        step  = 0

        while not done and step < args.max_steps:
            step      += 1
            time_step += 1

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action, logprob = agent.policy_old.act(state_t)

            a             = action.item()
            obs_next, r, done = env.step(ACTIONS[a], render=False)
            obs_queue.append(np.asarray(obs_next, dtype=np.float32))
            state_next    = np.concatenate(obs_queue)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(a)
            agent.buffer.logprobs.append(logprob.item())
            agent.buffer.rewards.append(float(r))
            agent.buffer.dones.append(done)

            ep_r  += float(r)
            state  = state_next

            if time_step % args.update_timestep == 0:
                agent.update()

        ep_rewards.append(ep_r)
        if ep_r > 500:
            successes += 1

        if (ep + 1) % 10 == 0:
            avg10   = np.mean(ep_rewards[-10:])
            elapsed = time.time() - t_total
            speed   = (ep + 1) / max(0.001, elapsed)
            remain  = (args.episodes - ep - 1) / max(0.001, speed) / 60
            print(f"  ep {ep+1:5d}/{args.episodes} | "
                  f"R={ep_r:8.1f} | "
                  f"Avg10={avg10:8.1f} | "
                  f"ok={successes:3d} | "
                  f"{speed:.2f} ep/s | "
                  f"~{remain:.1f} min left",
                  flush=True)

    print(f"\nTotal time: {(time.time()-t_total)/60:.1f} min")
    return agent, ep_rewards

# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",      type=str,   default="./obelix.py")
    ap.add_argument("--out",            type=str,   default="ppo_weights.pth")
    ap.add_argument("--episodes",       type=int,   default=675)
    ap.add_argument("--max_steps",      type=int,   default=400)
    ap.add_argument("--difficulty",     type=int,   default=3)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--update_timestep",type=int,   default=400)  # update every episode
    ap.add_argument("--K_epochs",       type=int,   default=4)
    ap.add_argument("--eps_clip",       type=float, default=0.2)
    ap.add_argument("--gamma",          type=float, default=0.99)
    ap.add_argument("--lr_actor",       type=float, default=0.0003)
    ap.add_argument("--lr_critic",      type=float, default=0.001)
    ap.add_argument("--entropy_coef",   type=float, default=0.01)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--wall_obstacles", action="store_true")
    args = ap.parse_args()

    OBELIX        = import_obelix(args.obelix_py)
    agent, rewards = train(OBELIX, args)

    # Save full network (actor + critic)
    torch.save({
        "actor":  agent.policy.actor.state_dict(),
        "critic": agent.policy.critic.state_dict(),
        "full":   agent.policy.state_dict(),
    }, args.out)
    print(f"Saved: {args.out}")

    # Plots
    w    = min(50, len(rewards))
    kern = np.ones(w) / w
    r    = [float(x) for x in rewards]
    avg  = np.convolve(r, kern, mode='valid')
    succ = np.convolve([1 if x > 500 else 0 for x in r], kern, mode='valid')
    eps  = np.arange(1, len(r) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(eps, r, alpha=0.3, color='steelblue', label='Reward')
    ax1.plot(eps[w-1:], avg, color='navy', linewidth=2, label=f'Avg({w})')
    ax1.set(xlabel='Episode', ylabel='Reward',
            title=f'PPO - Difficulty {args.difficulty}')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps[w-1:], succ * 100, color='green', linewidth=2)
    ax2.set(xlabel='Episode', ylabel='Success %',
            title='Success Rate', ylim=(0, 100))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ppo_plots.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()