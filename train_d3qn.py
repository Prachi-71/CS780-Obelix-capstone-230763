import numpy as np
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from obelix import OBELIX
except ImportError:
    print("ERROR: Run from CS780-OBELIX repo root.")
    sys.exit(1)

GAMMA         = 0.99
LR            = 0.001
BATCH_SIZE    = 64
BUFFER_SIZE   = 10000
TAU           = 0.005
ALPHA_PER     = 0.6
BETA_PER      = 0.4
BETA_RATE     = 0.001
UPDATE_FREQ   = 10
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.990
NO_EPISODES   = 800
MAX_STEPS     = 600

ACTIONS   = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS     = 18
device    = torch.device("cpu")

EXPLORE_BONUS = 0.5   # bonus for visiting new state

def shape_reward(r, obs, action, state, visit_counts):
    # Forward bonus: encourage moving forward
    if action == 2 and not obs[17]:
        r += 1.0
    # Exploration bonus: reward visiting new states
    if state not in visit_counts:
        r += EXPLORE_BONUS
    return r

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)

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

class D3QN_PER_Agent:
    def __init__(self):
        self.online_net = DuelingDQN(N_OBS, N_ACTIONS).to(device)
        self.target_net = DuelingDQN(N_OBS, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=LR)
        self.buffer     = PrioritizedReplayBuffer(BUFFER_SIZE, ALPHA_PER)
        self.epsilon    = EPSILON_START
        self.beta       = BETA_PER

    def select_action(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        with torch.no_grad():
            q = self.online_net(torch.FloatTensor(obs).unsqueeze(0))
        return int(q.argmax().item())

    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        idxs, weights, exps = self.buffer.sample(BATCH_SIZE, self.beta)
        states      = torch.FloatTensor([e[0] for e in exps])
        actions     = torch.LongTensor([e[1] for e in exps]).unsqueeze(1)
        rewards     = torch.FloatTensor([e[2] for e in exps]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in exps])
        dones       = torch.FloatTensor([e[4] for e in exps]).unsqueeze(1)

        with torch.no_grad():
            next_a  = self.online_net(next_states).argmax(1, keepdim=True)
            next_q  = self.target_net(next_states).gather(1, next_a)
            targets = rewards + GAMMA * next_q * (1 - dones)

        predicted = self.online_net(states).gather(1, actions)
        td_errors = (targets - predicted).abs().detach().squeeze(1).numpy()
        loss      = (weights.unsqueeze(1) * nn.HuberLoss(reduction='none')(predicted, targets)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()
        self.buffer.update(idxs, td_errors)

    def soft_update(self):
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(TAU * op.data + (1 - TAU) * tp.data)

def train():
    agent   = D3QN_PER_Agent()
    env     = OBELIX(scaling_factor=1, difficulty=0,
                     wall_obstacles=False, max_steps=MAX_STEPS)
    rewards   = []
    successes = 0
    t_total   = time.time()
    visit_counts = {}   # tracks visited states across all episodes

    print(f"\n{'='*60}")
    print(f"  D3QN-PER Optimized | {NO_EPISODES} eps | MAX_STEPS={MAX_STEPS}")
    print(f"  LR={LR} | BATCH={BATCH_SIZE} | EPS_DECAY={EPSILON_DECAY}")
    print(f"{'='*60}")

    for ep in range(NO_EPISODES):
        obs  = np.asarray(env.reset(), dtype=np.float32)
        done = False
        ep_r = 0.0
        step = 0

        while not done and step < MAX_STEPS:
            step    += 1
            a        = agent.select_action(obs)
            obs_next, r, done = env.step(ACTIONS[a], render=False)
            obs_next = np.asarray(obs_next, dtype=np.float32)
            ep_r    += r
            state_key = int(np.dot(obs.astype(np.int32), 1 << np.arange(18, dtype=np.int32)))
            r_shaped = shape_reward(r, obs, a, state_key, visit_counts)
            visit_counts[state_key] = visit_counts.get(state_key, 0) + 1
            agent.buffer.store((obs, a, r_shaped, obs_next, float(done)))
            agent.train_step()
            obs = obs_next

        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        agent.beta    = min(1.0, agent.beta + BETA_RATE)

        if (ep + 1) % UPDATE_FREQ == 0:
            agent.soft_update()

        rewards.append(ep_r)
        if ep_r > 500:
            successes += 1

        if (ep + 1) % 50 == 0 or (ep + 1) == NO_EPISODES:
            elapsed = time.time() - t_total
            speed   = (ep + 1) / max(0.001, elapsed)
            remain  = (NO_EPISODES - ep - 1) / max(0.001, speed) / 60
            print(f"  ep {ep+1:4d}/{NO_EPISODES} | "
                  f"eps={agent.epsilon:.3f} | "
                  f"R={ep_r:8.1f} | "
                  f"AvgR={np.mean(rewards[-50:]):8.1f} | "
                  f"ok={successes:3d} | "
                  f"states={len(visit_counts):5d} | "
                  f"{speed:.2f} ep/s | "
                  f"~{remain:.1f} min left",
                  flush=True)

    torch.save(agent.online_net.state_dict(), "d3qn_weights.pth")
    print(f"\nTotal time : {(time.time()-t_total)/60:.1f} min")
    print(f"Saved: d3qn_weights.pth")
    return rewards, agent


if __name__ == "__main__":
    rewards, agent = train()

    w    = 50
    kern = np.ones(w) / w
    rewards = [float(r) for r in rewards if r is not None]
    avg  = np.convolve(rewards, kern, mode='valid')
    succ = np.convolve([1 if r > 500 else 0 for r in rewards], kern, mode='valid')
    eps  = np.arange(1, len(rewards) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(eps, rewards, alpha=0.3, color='steelblue', label='Reward')
    ax1.plot(eps[w-1:], avg, color='navy', linewidth=2, label='Avg(50)')
    ax1.set(xlabel='Episode', ylabel='Reward', title='D3QN-PER Reward vs Episodes')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(eps[w-1:], succ * 100, color='green', linewidth=2)
    ax2.set(xlabel='Episode', ylabel='Success %', title='Success Rate vs Episodes', ylim=(0,100))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('d3qn_plots.png', dpi=150)
    plt.show()