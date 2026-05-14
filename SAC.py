import argparse
import time
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS = 18
STACK_SIZE = 4
N_OBS_STACKED = N_OBS * STACK_SIZE

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        probs = self.net(state)
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)
        return probs, log_probs

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.q1(state), self.q2(state)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

def import_obelix(obelix_py: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX

class SACAgent:
    def __init__(self, OBELIX, args):
        self.OBELIX = OBELIX
        self.args = args
        
        self.actor = Actor(N_OBS_STACKED, N_ACTIONS).to(device)
        self.critic = Critic(N_OBS_STACKED, N_ACTIONS).to(device)
        self.critic_target = Critic(N_OBS_STACKED, N_ACTIONS).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr)
        
        self.target_entropy = -np.log(1.0 / N_ACTIONS) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.lr)
        
        self.memory = ReplayBuffer(args.buffer_size)

    def train_step(self):
        if len(self.memory) < self.args.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.args.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_states)
            next_q1, next_q2 = self.critic_target(next_states)
            next_q = next_probs * (torch.min(next_q1, next_q2) - alpha * next_log_probs)
            next_q = next_q.sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.args.gamma * next_q

        current_q1, current_q2 = self.critic(states)
        current_q1 = current_q1.gather(1, actions)
        current_q2 = current_q2.gather(1, actions)
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        probs, log_probs = self.actor(states)
        q1, q2 = self.critic(states)
        min_q = torch.min(q1, q2)
        
        actor_loss = (probs * (alpha * log_probs - min_q.detach())).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach() * probs.detach()).sum(dim=1).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def run(self):
        print(f"Starting Fast Discrete SAC on {device.type.upper()} | Target: Diff 3")
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
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    probs, _ = self.actor(state_tensor)
                    action_idx = torch.multinomial(probs, 1).item()
                        
                next_obs_raw, reward, done = env.step(ACTIONS[action_idx], render=False)
                
                obs_queue.append(np.asarray(next_obs_raw, dtype=np.float32))
                next_state = np.concatenate(obs_queue)
                
                self.memory.add(state, action_idx, reward, next_state, done)
                
                # THE 4X SPEED HACK: Update only every 4 frames
                if step % 4 == 0:
                    self.train_step()
                
                state = next_state
                episode_reward += float(reward)
                step += 1
                
            ep_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_10 = np.mean(ep_rewards[-10:])
                elapsed = time.time() - start_time
                ep_sec = (episode + 1) / elapsed
                rem_min = (self.args.episodes - episode - 1) / ep_sec / 60
                
                current_alpha = self.log_alpha.exp().item()
                print(f"Ep {episode+1:4d}/{self.args.episodes} | Avg(10): {avg_10:7.1f} | Alpha: {current_alpha:.3f} | ~{rem_min:.1f}m left")

        return self.actor, ep_rewards

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py", type=str, default="./obelix.py")
    ap.add_argument("--out", type=str, default="sac_diff3_weights.pth")
    # Reduced episodes and increased LR for the 2-hour speed run
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--wall_obstacles", action="store_true", default=True)
    ap.add_argument("--scaling_factor", type=int, default=5)
    
    ap.add_argument("--lr", type=float, default=0.0005)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--buffer_size", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()

    OBELIX = import_obelix(args.obelix_py)
    agent = SACAgent(OBELIX, args)
    
    trained_actor, rewards = agent.run()

    torch.save(trained_actor.state_dict(), args.out)
    print(f"\nSaved SAC Actor weights to: {args.out}")

    print("Generating learning curve plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, color='#1f77b4', alpha=0.3, label='Raw Episode Reward')
    
    if len(rewards) >= 100:
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        plt.plot(np.arange(len(moving_avg)) + 99, moving_avg, color='red', linewidth=2, label='100-Episode Moving Avg')

    plt.title("SAC Training Curve - Difficulty 3", fontsize=14, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    plot_filename = "sac_learning_curve.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved training plot to: {plot_filename}")

if __name__ == "__main__":
    main()