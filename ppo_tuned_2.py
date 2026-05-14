"""
PPO Fine-Tuning Script for OBELIX
===================================
Loads your existing ppo_weights.pth (trained on diff 3)
and fine-tunes via curriculum: diff 0 → diff 1 → diff 2 → diff 3
Features: Bulldozer + Anti-Spin Reward Shaping & Reward Scaling

Run:
    python finetune_ppo.py

Produces: ppo_finetuned.pth  (drop-in replacement for ppo_weights.pth)
"""

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

ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS     = 5
N_OBS         = 18
STACK_SIZE    = 4
N_OBS_STACKED = N_OBS * STACK_SIZE  # 72

# ── Curriculum schedule ─────────────────────────────────────────────────
# Tune episodes per level based on time budget:
# ~300 eps diff0 = ~20 min, ~300 eps diff1 = ~25 min,
# ~200 eps diff2 = ~25 min, ~200 eps diff3 = ~25 min
# Total: ~95 min — fits comfortably in 6 hours
CURRICULUM = [
    {"difficulty": 0, "episodes": 300, "max_steps": 400, "lr_actor": 1e-4, "lr_critic": 3e-4},
    {"difficulty": 1, "episodes": 300, "max_steps": 400, "lr_actor": 5e-5, "lr_critic": 1e-4},
    {"difficulty": 2, "episodes": 200, "max_steps": 400, "lr_actor": 3e-5, "lr_critic": 8e-5},
    {"difficulty": 3, "episodes": 200, "max_steps": 400, "lr_actor": 1e-5, "lr_critic": 3e-5},
]

# PPO hyperparameters
GAMMA         = 0.99
K_EPOCHS      = 4
EPS_CLIP      = 0.2
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 400     # steps between PPO updates
SCALING       = 5
SEED          = 42

WEIGHTS_IN    = "ppo_weights.pth"
WEIGHTS_OUT   = "ppo_finetuned.pth"


# ── Same architecture as your original ppo.py ───────────────────────────
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128), nn.Tanh(),
            nn.Linear(128, 64),        nn.Tanh(),
            nn.Linear(64, 1),
        )

    def act(self, state):
        probs  = self.actor(state)
        dist   = Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        probs        = self.actor(state)
        dist         = Categorical(probs)
        logprobs     = dist.log_prob(action)
        entropy      = dist.entropy()
        state_values = self.critic(state)
        return logprobs, state_values, entropy


# ── Rollout buffer ───────────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self):
        self.states = []; self.actions = []; self.logprobs = []
        self.rewards = []; self.dones = []

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ── Load weights ─────────────────────────────────────────────────────────
def load_weights(model, path):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "full" in ckpt:
        model.load_state_dict(ckpt["full"])
        print(f"  Loaded full state dict from {path}")
    elif "actor" in ckpt and "critic" in ckpt:
        model.actor.load_state_dict(ckpt["actor"])
        model.critic.load_state_dict(ckpt["critic"])
        print(f"  Loaded actor+critic separately from {path}")
    else:
        model.load_state_dict(ckpt)
        print(f"  Loaded raw state dict from {path}")


# ── PPO update ───────────────────────────────────────────────────────────
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

    old_states   = torch.FloatTensor(np.array(buffer.states))
    old_actions  = torch.LongTensor(np.array(buffer.actions))
    old_logprobs = torch.FloatTensor(np.array(buffer.logprobs))

    for _ in range(K_EPOCHS):
        logprobs, values, entropy = policy.evaluate(old_states, old_actions)
        values     = values.squeeze()
        ratios     = torch.exp(logprobs - old_logprobs)
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


# ── Fine-tuning loop ─────────────────────────────────────────────────────
def finetune():
    try:
        from obelix import OBELIX
    except ImportError:
        raise ImportError("obelix.py must be in the same directory.")

    # Load model
    policy     = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)
    policy_old = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)

    import os
    if os.path.exists(WEIGHTS_IN):
        load_weights(policy, WEIGHTS_IN)
    else:
        print(f"  WARNING: {WEIGHTS_IN} not found — training from scratch")

    policy_old.load_state_dict(policy.state_dict())
    mse = nn.MSELoss()

    all_rewards  = []
    all_diffs    = []
    best_reward  = -np.inf
    t_total      = time.time()

    for stage in CURRICULUM:
        diff     = stage["difficulty"]
        n_eps    = stage["episodes"]
        max_s    = stage["max_steps"]
        lr_a     = stage["lr_actor"]
        lr_c     = stage["lr_critic"]

        # New optimizer with lower LR for each stage (conservative fine-tuning)
        optimizer = optim.Adam([
            {"params": policy.actor.parameters(),  "lr": lr_a},
            {"params": policy.critic.parameters(), "lr": lr_c},
        ])

        buffer    = RolloutBuffer()
        time_step = 0
        successes = 0

        print(f"\n{'='*60}")
        print(f"  Fine-tuning: Difficulty {diff} | {n_eps} episodes")
        print(f"  lr_actor={lr_a} | lr_critic={lr_c}")
        print(f"{'='*60}")

        stage_rewards = []

        for ep in range(n_eps):
            env = OBELIX(
                scaling_factor=SCALING,
                max_steps=max_s,
                wall_obstacles=True,
                difficulty=diff,
                seed=SEED + ep,
            )

            try:
                obs_raw = np.asarray(env.reset(seed=SEED + ep), dtype=np.float32)
            except TypeError:
                obs_raw = np.asarray(env.reset(), dtype=np.float32)

            obs_queue = deque([obs_raw] * STACK_SIZE, maxlen=STACK_SIZE)
            state     = np.concatenate(obs_queue)

            ep_r = 0.0
            done = False
            step = 0
            has_box = False # Reset box tracker every episode

            while not done and step < max_s:
                step      += 1
                time_step += 1

                state_t = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action, logprob = policy_old.act(state_t)

                a = action.item()
                obs_next, r, done = env.step(ACTIONS[a])
                obs_queue.append(np.asarray(obs_next, dtype=np.float32))
                state_next = np.concatenate(obs_queue)

                # ── Reward shaping ──────────────────────────────────────────
                raw_r      = float(r)
                shaped_r   = raw_r
                action_str = ACTIONS[a]
                stuck_flag = int(obs_next[17])

                if raw_r >= 90:
                    has_box = True

                if has_box and not done:
                    # BULLDOZER MODE
                    if action_str == "FW" and stuck_flag == 0:
                        shaped_r += 10.0
                    elif action_str in ["L45", "R45", "L22", "R22"]:
                        shaped_r -= 5.0
                elif not has_box and not done:
                    # ANTI-SPIN / EXPLORATION MODE
                    if action_str in ["L45", "R45"]:
                        shaped_r -= 2.5
                    elif action_str == "FW" and stuck_flag == 0:
                        shaped_r += 0.5

                shaped_r = shaped_r / 100.0
                # ────────────────────────────────────────────────────────────

                buffer.states.append(state)
                buffer.actions.append(a)
                buffer.logprobs.append(logprob.item())
                buffer.rewards.append(shaped_r) # Push shaped, scaled reward to PPO memory
                buffer.dones.append(done)

                ep_r  += raw_r # Keep logging the raw, unshaped reward for accurate printing
                state  = state_next

                if time_step % UPDATE_EVERY == 0:
                    ppo_update(policy, policy_old, optimizer, buffer, mse)

            try:
                env.close()
            except Exception:
                pass

            stage_rewards.append(ep_r)
            all_rewards.append(ep_r)
            all_diffs.append(diff)

            if ep_r > 500:
                successes += 1

            # Save best weights
            mean10 = np.mean(stage_rewards[-10:])
            if mean10 > best_reward:
                best_reward = mean10
                torch.save({
                    "actor":  policy.actor.state_dict(),
                    "critic": policy.critic.state_dict(),
                    "full":   policy.state_dict(),
                }, WEIGHTS_OUT)

            if (ep + 1) % 20 == 0:
                mean20  = np.mean(stage_rewards[-20:])
                elapsed = time.time() - t_total
                speed   = (len(all_rewards)) / max(0.001, elapsed)
                print(f"  [D{diff} ep {ep+1:>4}/{n_eps}]  "
                      f"R={ep_r:>8.1f}  Avg20={mean20:>8.1f}  "
                      f"ok={successes}  {speed:.2f} ep/s")

        print(f"  Stage D{diff} done | "
              f"best_mean10={best_reward:.1f} | "
              f"successes={successes}/{n_eps}")

    print(f"\nTotal fine-tuning time: {(time.time()-t_total)/60:.1f} min")
    print(f"Best weights saved to: {WEIGHTS_OUT}")

    # Plot
    _plot(all_rewards, all_diffs)


def _plot(rewards, diffs):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
    labels = {0: "Diff 0", 1: "Diff 1", 2: "Diff 2", 3: "Diff 3"}

    x = np.arange(len(rewards))
    for d in sorted(set(diffs)):
        idx = [i for i, v in enumerate(diffs) if v == d]
        ax.scatter(idx, [rewards[i] for i in idx],
                   alpha=0.2, s=6, color=colors[d])

    w    = min(30, len(rewards))
    kern = np.ones(w) / w
    avg  = np.convolve(rewards, kern, mode="valid")
    ax.plot(np.arange(w-1, len(rewards)), avg,
            color="black", linewidth=2, label="Avg(30)")

    # Difficulty boundary markers
    counts = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    boundary = 0
    for d in sorted(set(diffs))[:-1]:
        boundary += counts[d]
        ax.axvline(boundary, color=colors[d], linestyle="--",
                   alpha=0.5, label=f"→ D{d+1}")

    from matplotlib.patches import Patch
    legend = [Patch(color=colors[d], label=labels[d]) for d in sorted(set(diffs))]
    ax.legend(handles=legend + [plt.Line2D([0],[0],color="black",lw=2,label="Avg(30)")],
              loc="lower right")
    ax.set(title="PPO Fine-Tuning Curriculum", xlabel="Episode", ylabel="Reward")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ppo_finetune_plots_1.png", dpi=150)
    print("Plot saved: ppo_finetune_plots_1.png")


if __name__ == "__main__":
    finetune()