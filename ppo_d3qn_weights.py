"""
pretrain_ppo_from_d3qn.py — Behavioral Cloning: D3QN → PPO
============================================================
Since D3QN and PPO have different architectures, direct weight transfer
is impossible. Instead, we:

  1. Run the trained D3QN agent to collect expert (state, action) demos
  2. Pre-train the PPO Actor via supervised cross-entropy loss (BC)
  3. Save a ppo_weights_bc.pth ready for finetune_ppo.py

Pipeline:
  d3qn_weights.pth  →  demo collection  →  BC training  →  ppo_weights_bc.pth
                                                               ↓
                                                       finetune_ppo.py

Usage:
  python pretrain_ppo_from_d3qn.py

Then in finetune_ppo.py set:
  WEIGHTS_IN = "ppo_weights_bc.pth"
"""

# ── Fix Qt/OpenCV display crash (must be FIRST) ──────────────────────────
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cpu")

# ── Shared constants ─────────────────────────────────────────────────────
ACTIONS       = ["L45", "L22", "FW", "R22", "R45"]
N_ACTIONS     = 5
N_OBS         = 18
STACK_SIZE    = 4
N_OBS_STACKED = N_OBS * STACK_SIZE   # 72

D3QN_WEIGHTS  = "d3qn_weights_backup.pth"   # ← your D3QN PER weights file
PPO_WEIGHTS   = "ppo_weights_bc.pth" # ← output: use as WEIGHTS_IN in finetune_ppo.py

SCALING       = 5
SEED          = 42

# Demo collection settings
DEMO_EPISODES_PER_DIFF = 150   # episodes of D3QN gameplay to record per difficulty
MAX_STEPS              = 400

# BC training settings
BC_EPOCHS     = 30             # supervised training epochs over demo dataset
BC_LR         = 3e-4
BC_BATCH_SIZE = 256


# ── FrameStacker (shared by both D3QN demo collection and PPO) ───────────
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

    def get_state(self) -> np.ndarray:
        return np.concatenate(list(self.frames))


# ════════════════════════════════════════════════════════════════════════
# PART 1 — D3QN Architecture (for loading weights and generating demos)
# ════════════════════════════════════════════════════════════════════════

HIDDEN1      = 256
HIDDEN2      = 128
VALUE_HIDDEN = 64
ADV_HIDDEN   = 64

class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(N_OBS_STACKED, HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1, HIDDEN2),        nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(HIDDEN2, VALUE_HIDDEN), nn.ReLU(),
            nn.Linear(VALUE_HIDDEN, 1),
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(HIDDEN2, ADV_HIDDEN), nn.ReLU(),
            nn.Linear(ADV_HIDDEN, N_ACTIONS),
        )

    def forward(self, x):
        feat = self.backbone(x)
        V    = self.value_stream(feat)
        A    = self.adv_stream(feat)
        return V + (A - A.mean(dim=1, keepdim=True))


def load_d3qn(path: str) -> DuelingDQN:
    net  = DuelingDQN().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    key  = "online" if "online" in ckpt else list(ckpt.keys())[0]
    net.load_state_dict(ckpt[key])
    net.eval()
    print(f"  D3QN loaded from '{path}' (key='{key}')")
    return net


# ════════════════════════════════════════════════════════════════════════
# PART 2 — PPO Actor-Critic (target architecture to pre-train)
# ════════════════════════════════════════════════════════════════════════

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
        dist   = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state, action):
        probs        = self.actor(state)
        dist         = torch.distributions.Categorical(probs)
        logprobs     = dist.log_prob(action)
        entropy      = dist.entropy()
        state_values = self.critic(state)
        return logprobs, state_values, entropy


# ════════════════════════════════════════════════════════════════════════
# PART 3 — Demo Collection (run D3QN, record state→action pairs)
# ════════════════════════════════════════════════════════════════════════

def collect_demos(d3qn_net: DuelingDQN, env_factory, difficulty: int,
                  n_episodes: int, label: str):
    """
    Run D3QN greedily and record every (state, action_idx) pair.
    Returns arrays: states (N, 72), actions (N,)
    """
    all_states  = []
    all_actions = []
    stacker     = FrameStacker()
    total_reward = 0.0
    successes    = 0

    for ep in range(n_episodes):
        print(f"\r  [{label}] Collecting demo {ep+1}/{n_episodes}...", end="", flush=True)

        env = env_factory()
        try:
            obs = env.reset(seed=SEED + ep)
        except TypeError:
            obs = env.reset()

        stacker.reset()
        stacker.push(obs)
        state = stacker.get_state()

        ep_r = 0.0
        done = False

        for _ in range(MAX_STEPS):
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_vals = d3qn_net(state_t)
                action_idx = q_vals.argmax(dim=1).item()

            # Record (state, action) BEFORE stepping
            all_states.append(state.copy())
            all_actions.append(action_idx)

            obs_next, reward, done = env.step(ACTIONS[action_idx])
            stacker.push(obs_next)
            state = stacker.get_state()
            ep_r += float(reward)

            if done:
                break

        total_reward += ep_r
        if ep_r > 500:
            successes += 1

        try:
            env.close()
        except Exception:
            pass

    print()
    avg_r = total_reward / n_episodes
    print(f"  [{label}] Done | avg_reward={avg_r:.1f} | "
          f"successes={successes}/{n_episodes} | "
          f"demo_steps={len(all_states)}")

    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Behavioral Cloning (supervised imitation of D3QN)
# ════════════════════════════════════════════════════════════════════════

def behavioral_cloning(policy: ActorCritic,
                        states: np.ndarray,
                        actions: np.ndarray) -> list:
    """
    Train the PPO Actor to predict D3QN's chosen actions.
    Loss: cross-entropy between actor's action distribution and D3QN's greedy action.
    The Critic is left randomly initialised — PPO will learn it during fine-tuning.
    """
    optimizer = optim.Adam(policy.actor.parameters(), lr=BC_LR)
    ce_loss   = nn.CrossEntropyLoss()

    n_samples = len(states)
    n_batches = max(1, n_samples // BC_BATCH_SIZE)

    losses = []
    accs   = []

    print(f"\n  BC training: {n_samples} demo steps | "
          f"{BC_EPOCHS} epochs | batch={BC_BATCH_SIZE}")

    for epoch in range(BC_EPOCHS):
        # Shuffle dataset each epoch
        perm = np.random.permutation(n_samples)
        s_shuf = states[perm]
        a_shuf = actions[perm]

        epoch_loss = 0.0
        epoch_correct = 0

        for b in range(n_batches):
            start = b * BC_BATCH_SIZE
            end   = min(start + BC_BATCH_SIZE, n_samples)

            s_batch = torch.FloatTensor(s_shuf[start:end]).to(device)
            a_batch = torch.LongTensor(a_shuf[start:end]).to(device)

            # Actor outputs softmax probabilities → use logits for cross-entropy
            # We need raw logits: remove the Softmax and pass through manually
            # Instead, use NLLLoss on log of actor probabilities (numerically stable)
            probs    = policy.actor(s_batch)          # softmax probs
            log_probs = torch.log(probs + 1e-8)       # log probs
            loss     = nn.functional.nll_loss(log_probs, a_batch)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.actor.parameters(), 1.0)
            optimizer.step()

            epoch_loss    += loss.item()
            epoch_correct += (probs.argmax(dim=1) == a_batch).sum().item()

        avg_loss = epoch_loss / n_batches
        accuracy = 100.0 * epoch_correct / n_samples
        losses.append(avg_loss)
        accs.append(accuracy)

        if (epoch + 1) % 5 == 0:
            print(f"  [BC Epoch {epoch+1:>3}/{BC_EPOCHS}]  "
                  f"loss={avg_loss:.4f}  accuracy={accuracy:.1f}%")

    print(f"  BC complete | final accuracy={accs[-1]:.1f}%")
    return losses, accs


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    try:
        from obelix import OBELIX
    except ImportError:
        raise ImportError("obelix.py must be in the same directory.")

    if not os.path.exists(D3QN_WEIGHTS):
        raise FileNotFoundError(
            f"D3QN weights not found: '{D3QN_WEIGHTS}'\n"
            f"Make sure your d3qn_weights.pth is in this directory."
        )

    # ── Step 1: Load D3QN ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 1: Loading D3QN expert")
    print("="*60)
    d3qn_net = load_d3qn(D3QN_WEIGHTS)

    # ── Step 2: Collect demonstrations across all 3 difficulties ────────
    print("\n" + "="*60)
    print("  STEP 2: Collecting D3QN expert demonstrations")
    print("="*60)

    # Mirrors D3QN curriculum: diff 0 (static), diff 2 (blinking), diff 3 (moving)
    demo_configs = [
        {"difficulty": 0, "label": "Diff 0 – Static"},
        {"difficulty": 2, "label": "Diff 2 – Blinking"},
        {"difficulty": 3, "label": "Diff 3 – Moving+Blinking"},
    ]

    all_demo_states  = []
    all_demo_actions = []

    for cfg in demo_configs:
        diff  = cfg["difficulty"]
        label = cfg["label"]

        def make_env(d=diff):
            return OBELIX(
                scaling_factor=SCALING,
                max_steps=MAX_STEPS,
                wall_obstacles=True,
                difficulty=d,
                seed=SEED,
            )

        s, a = collect_demos(d3qn_net, make_env, diff,
                             DEMO_EPISODES_PER_DIFF, label)
        all_demo_states.append(s)
        all_demo_actions.append(a)

    demo_states  = np.concatenate(all_demo_states,  axis=0)
    demo_actions = np.concatenate(all_demo_actions, axis=0)

    print(f"\n  Total demo dataset: {len(demo_states)} (state, action) pairs")

    # Class distribution — check D3QN isn't degenerate
    unique, counts = np.unique(demo_actions, return_counts=True)
    print("  Action distribution in demos:")
    for u, c in zip(unique, counts):
        print(f"    {ACTIONS[u]:>4}: {c:>6} ({100*c/len(demo_actions):.1f}%)")

    # ── Step 3: Build fresh PPO model ───────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 3: Behavioral Cloning (D3QN → PPO Actor)")
    print("="*60)

    policy = ActorCritic(N_OBS_STACKED, N_ACTIONS).to(device)

    # ── Step 4: BC training ─────────────────────────────────────────────
    losses, accs = behavioral_cloning(policy, demo_states, demo_actions)

    # ── Step 5: Save BC-initialised PPO weights ─────────────────────────
    print("\n" + "="*60)
    print("  STEP 4: Saving BC-initialised PPO weights")
    print("="*60)

    torch.save({
        "actor":  policy.actor.state_dict(),
        "critic": policy.critic.state_dict(),
        "full":   policy.state_dict(),
    }, PPO_WEIGHTS)

    print(f"  Saved: {PPO_WEIGHTS}")
    print(f"\n  Next step: open finetune_ppo.py and set")
    print(f"    WEIGHTS_IN = \"{PPO_WEIGHTS}\"")
    print(f"  Then run: python finetune_ppo.py")

    # ── Plot BC learning curve ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, color="#4C72B0", linewidth=2)
    ax1.set_title("BC Training Loss (Cross-Entropy)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(accs, color="#55A868", linewidth=2)
    ax2.set_title("BC Action Accuracy (vs D3QN)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.axhline(100 / N_ACTIONS, color="red", linestyle="--",
                alpha=0.5, label=f"Random baseline ({100/N_ACTIONS:.0f}%)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("PPO Behavioral Cloning from D3QN Expert", fontsize=13)
    plt.tight_layout()
    plt.savefig("bc_training_curve.png", dpi=150)
    print("  Plot saved: bc_training_curve.png")


if __name__ == "__main__":
    main()