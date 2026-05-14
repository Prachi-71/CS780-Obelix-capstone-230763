import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cpu")

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS = 5
N_OBS = 18
STACK_SIZE = 4
N_OBS_STACKED = N_OBS * STACK_SIZE

CURRICULUM = [
    {"difficulty": 0, "episodes": 300, "max_steps": 400, "lr_actor": 1e-4, "lr_critic": 3e-4},
    {"difficulty": 2, "episodes": 300, "max_steps": 400, "lr_actor": 5e-5, "lr_critic": 1e-4},
    {"difficulty": 3, "episodes": 200, "max_steps": 400, "lr_actor": 1e-5, "lr_critic": 3e-5},
]

GAMMA = 0.99
K_EPOCHS = 4
EPS_CLIP = 0.2
ENTROPY_COEF = 0.01
UPDATE_EVERY = 400
SCALING = 5
SEED = 42

WEIGHTS_IN = "ppo_weights_bc.pth"
WEIGHTS_OUT = "ppo_finetuned_2.pth"


class FrameStacker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = deque(
            [np.zeros(N_OBS, dtype=np.float32)] * STACK_SIZE,
            maxlen=STACK_SIZE,
        )

    def push(self, obs):
        self.frames.append(np.asarray(obs, dtype=np.float32))

    def get_state(self):
        return np.concatenate(list(self.frames))


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def act(self, state):
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        probs = self.actor(state)
        dist = Categorical(probs)
        logprobs = dist.log_prob(action)
        entropy = dist.entropy()
        state_values = self.critic(state)
        return logprobs, state_values, entropy


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


def load_weights(model, path):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "full" in ckpt:
        model.load_state_dict(ckpt["full"])
    elif "actor" in ckpt and "critic" in ckpt:
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
    else:
        model.load_state_dict(ckpt)


def ppo_update(policy, policy_old, optimizer, buffer, mse):
    returns = []
    G = 0
    for r, done in zip(reversed(buffer.rewards), reversed(buffer.dones)):
        if done:
            G = 0
        G = r + GAMMA * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-7)

    old_states = torch.FloatTensor(np.array(buffer.states))
    old_actions = torch.LongTensor(np.array(buffer.actions))
    old_logprobs = torch.FloatTensor(np.array(buffer.logprobs))

    for _ in range(K_EPOCHS):
        logprobs, values, entropy = policy.evaluate(old_states, old_actions)
        values = values.squeeze()
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

        loss = (-torch.min(surr1, surr2)
                + 0.5 * mse(values, returns)
                - ENTROPY_COEF * entropy).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    policy_old.load_state_dict(policy.state_dict())
    buffer.clear()


def shape_reward(raw_r, action_str, stuck_flag, has_box, done):
    shaped = float(raw_r)

    if has_box and not done:
        if action_str == "FW" and stuck_flag == 0:
            shaped += 10.0
        elif action_str in ["L45", "R45", "L22", "R22"]:
            shaped -= 5.0
        shaped /= 100.0

    elif not has_box and not done:
        if action_str in ["L45", "R45", "L22", "R22"]:
            shaped -= 15
        elif action_str == "FW" and stuck_flag == 0:
            shaped += 20
        shaped /= 100.0

    else:
        shaped /= 100.0

    return shaped


def finetune():
    from obelix import OBELIX

    policy = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)
    policy_old = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)

    if os.path.exists(WEIGHTS_IN):
        load_weights(policy, WEIGHTS_IN)

    policy_old.load_state_dict(policy.state_dict())
    mse = nn.MSELoss()

    all_rewards = []
    all_diffs = []
    best_reward = -np.inf
    t_total = time.time()

    for stage in CURRICULUM:
        diff = stage["difficulty"]
        n_eps = stage["episodes"]
        max_s = stage["max_steps"]
        lr_a = stage["lr_actor"]
        lr_c = stage["lr_critic"]

        optimizer = optim.Adam([
            {"params": policy.actor.parameters(), "lr": lr_a},
            {"params": policy.critic.parameters(), "lr": lr_c},
        ])

        buffer = RolloutBuffer()
        stacker = FrameStacker()
        time_step = 0
        successes = 0
        stage_rewards = []

        for ep in range(n_eps):
            env = OBELIX(
                scaling_factor=SCALING,
                max_steps=max_s,
                wall_obstacles=True,
                difficulty=diff,
                seed=SEED + ep,
            )

            stacker.reset()
            obs_raw = env.reset()
            stacker.push(obs_raw)
            state = stacker.get_state()

            ep_r = 0.0
            done = False
            step = 0
            has_box = False

            while not done and step < max_s:
                step += 1
                time_step += 1

                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, logprob = policy_old.act(state_t)

                a = action.item()
                action_str = ACTIONS[a]

                obs_next, r, done = env.step(action_str)
                stacker.push(obs_next)
                state_next = stacker.get_state()

                raw_r = float(r)
                stuck_flag = int(obs_next[17])

                if raw_r >= 90:
                    has_box = True

                shaped_r = shape_reward(raw_r, action_str, stuck_flag, has_box, done)

                buffer.states.append(state)
                buffer.actions.append(a)
                buffer.logprobs.append(logprob.item())
                buffer.rewards.append(shaped_r)
                buffer.dones.append(done)

                ep_r += raw_r
                state = state_next

                if time_step % UPDATE_EVERY == 0:
                    ppo_update(policy, policy_old, optimizer, buffer, mse)

            env.close()

            stage_rewards.append(ep_r)
            all_rewards.append(ep_r)
            all_diffs.append(diff)

            if ep_r > 500:
                successes += 1

            mean10 = np.mean(stage_rewards[-10:])
            if mean10 > best_reward:
                best_reward = mean10
                torch.save({
                    "actor": policy.actor.state_dict(),
                    "critic": policy.critic.state_dict(),
                    "full": policy.state_dict(),
                }, WEIGHTS_OUT)

    print(f"Saved to {WEIGHTS_OUT}")


if __name__ == "__main__":
    finetune()