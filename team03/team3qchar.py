import math
import random
from collections import deque
import copy

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from entity import CharacterEntity
from events import Event

class GridDQN(nn.Module):
    def __init__(self, in_ch=6, patch=9, n_actions=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        flat = 32 * patch * patch
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)
    def push(self, *transition):
        self.buf.append(tuple(transition))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d, dtype=torch.bool),
        )
    def __len__(self):
        return len(self.buf)

class QChar(CharacterEntity):
    last_bombed = False

    ACTIONS = [
        ("stay",   (0,  0), False),
        ("up",     (0, -1), False),
        ("down",   (0,  1), False),
        ("left",   (-1, 0), False),
        ("right",  (1,  0), False),
        ("bomb",   (0,  0), True),
    ]

    def __init__(
        self,
        *args, 
        radius=4,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=20000,
        target_update_every=1000,
        buffer_capacity=50000,
        train_every=4,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.radius = radius
        patch = 2 * radius + 1
        self.n_actions = len(self.ACTIONS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = GridDQN(in_ch=6, patch=patch, n_actions=self.n_actions).to(self.device)
        self.target = GridDQN(in_ch=6, patch=patch, n_actions=self.n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_every = train_every

        self.epsilon = epsilon_start
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        self.target_update_every = target_update_every

        self._alive_last_tick = True
        self.last_bomb_step = -100  # Track when we last bombed
        self.episode_steps = 0  # Track steps within episode
        self.episode_reward = 0.0  # Track cumulative reward

    def do(self, wrld):
        # Current state features
        state = self._featurize(wrld).to(self.device)

        # Choose action using one-step look-ahead; also get simulated next info
        (a_idx,
         sim_next_world,
         sim_events,
         reward,
         done,
         next_state_feat) = self._choose_and_simulate(wrld, state)

        # Store transition immediately so we don't lose terminal steps
        self.buffer.push(
            state.detach().cpu(),
            a_idx,
            reward,
            next_state_feat.detach().cpu(),
            done
        )

        # Train
        if (self.total_steps % self.train_every) == 0 and len(self.buffer) >= self.batch_size:
            self._optimize()

        # Periodic auto-save during training
        if self.total_steps > 0 and self.total_steps % 500 == 0:
            self.save_model('qchar_model_latest.pt')

        # Execute the chosen action in the real world
        name, (dx, dy), bomb = self.ACTIONS[a_idx]
        if bomb:
            try:
                self.place_bomb()
                self.last_bomb_step = self.total_steps
            except Exception:
                pass
        else:
            self.move(dx, dy)

        # Bookkeeping
        self.total_steps += 1
        self.episode_steps += 1
        self.episode_reward += reward
        self._update_epsilon()
        if self.total_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

    def reset_episode(self):
        """Call this at the start of each new game/episode"""
        if self.episode_steps > 0:
            print(f"Episode finished: Steps={self.episode_steps}, Reward={self.episode_reward:.2f}, Epsilon={self.epsilon:.3f}")
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.last_bomb_step = self.total_steps - 100  # Reset bomb cooldown

    # ---------- DQN optimize ----------
    def _optimize(self):
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Q(s,a)
        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]
            target = rewards + (~dones).float() * self.gamma * next_q

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()

    # ---------- Action selection w/ look-ahead safety mask ----------
    def _choose_and_simulate(self, wrld, state_t):
        sims = []
        safe_idxs = []

        for i in range(self.n_actions):
            sim_w2, sim_events, me_alive_after, next_feat = self._simulate_action(wrld, i)
            r, d = self._lookahead_reward(wrld, sim_w2, sim_events, i)
            sims.append((sim_w2, sim_events, r, d, next_feat, me_alive_after))
            if me_alive_after:
                safe_idxs.append(i)

        if random.random() < self.epsilon:
            if safe_idxs:
                a_idx = random.choice(safe_idxs)
            else:
                a_idx = random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                q = self.policy(state_t.unsqueeze(0)).squeeze(0)  # (A,)
                if safe_idxs:
                    mask = torch.full_like(q, float('-inf'))
                    mask[safe_idxs] = 0.0
                    q = q + mask
                a_idx = int(q.argmax().item())

        sim_next_world, sim_events, reward, done, next_feat, _ = sims[a_idx]
        return a_idx, sim_next_world, sim_events, reward, done, next_feat

    def _simulate_action(self, wrld, a_idx):
        wcopy = self._copy_world(wrld)
        me_clone = self._find_me(wcopy)
        name, (dx, dy), bomb = self.ACTIONS[a_idx]

        if me_clone is not None:
            if bomb:
                try:
                    me_clone.place_bomb()
                except Exception:
                    pass
            else:
                try:
                    me_clone.move(dx, dy)
                except Exception:
                    pass

        w2, events = self._world_next(wcopy)
        me_after = self._find_me(w2)
        next_feat = self._featurize(w2)
        me_alive_after = me_after is not None
        return w2, events, me_alive_after, next_feat

    def _world_next(self, w):
        try:
            out = w.next()
        except Exception:
            return w, getattr(w, "events", [])
        if isinstance(out, tuple) and len(out) == 2:
            nw, ev = out
        else:
            nw = out
            ev = getattr(nw, "events", [])
        ev = ev if ev is not None else []
        return nw, ev

    # ---------- Reward from simulated events ----------
    def _lookahead_reward(self, w_before, w_after, events, action_idx):
        reward = 0.0
        done = False

        me_before = self._find_me(w_before)
        me_after  = self._find_me(w_after)

        # Parse events caused by the simulated tick
        for e in events:
            if e.tpe == Event.CHARACTER_KILLED_BY_MONSTER and self._same_char(e.character):
                reward -= 10.0
                done = True
            elif e.tpe == Event.CHARACTER_FOUND_EXIT and self._same_char(e.character):
                print("Exit Found!")
                reward += 100.0
                done = True
            elif e.tpe == Event.BOMB_HIT_CHARACTER and self._same_char(e.other):
                reward -= 15.0  # Heavy penalty for getting hit by bomb
                done = True

        # If we simply don't exist after the tick, it's terminal death
        if me_before is not None and me_after is None:
            reward -= 50.0
            done = True

        # CRITICAL: Penalize being in danger zones
        if me_after is not None and not done:
            if self._is_tile_dangerous(w_after, me_after.x, me_after.y):
                reward -= 8.0  # Strong penalty for being in danger
            
            # Additional penalty for staying near bombs
            danger_proximity = self._get_danger_proximity(w_after, me_after.x, me_after.y)
            reward -= danger_proximity * 2.0

        # Discourage excessive bombing
        name, (dx, dy), is_bomb = self.ACTIONS[action_idx]
        if is_bomb:
            # Penalize bombing if we just bombed recently
            if self.total_steps - self.last_bomb_step < 10:
                reward -= 5.0
            if me_after is not None:
                escape_routes = self._count_escape_routes(w_after, me_after.x, me_after.y)
                if escape_routes < 2:
                    reward -= 10.0  # Don't bomb if trapped

        # Shaping: distance to exit (before -> after)
        ex = getattr(w_before, "exitcell", None)
        if ex is not None and me_before is not None and me_after is not None:
            exx, exy = ex
            d0 = abs(me_before.x - exx) + abs(me_before.y - exy)
            d1 = abs(me_after.x - exx) + abs(me_after.y - exy)
            delta = d0 - d1
            # Reward for moving closer, penalize for moving away
            reward += 0.5 * delta

        # Small negative reward for staying still (encourage exploration)
        if action_idx == 0:  # stay action
            reward -= 0.1

        return reward, done

    def _get_danger_proximity(self, wrld, x, y):
        """Calculate how close we are to danger (0 = safe, higher = more dangerous)"""
        danger_score = 0.0
        try:
            rng = wrld.expl_range
        except Exception:
            rng = 3
        
        for b in getattr(wrld, "bombs", {}).values():
            bx, by = b.x, b.y
            dist = abs(x - bx) + abs(y - by)
            if dist <= rng:
                # Closer bombs and bombs about to explode are more dangerous
                time_factor = (4 - b.timer) / 4.0  # 0 to 1, higher when about to explode
                dist_factor = (rng - dist + 1) / (rng + 1)  # 0 to 1, higher when closer
                danger_score += time_factor * dist_factor
        
        return danger_score

    def _count_escape_routes(self, wrld, x, y):
        """Count number of safe directions from current position"""
        try:
            rng = wrld.expl_range
        except Exception:
            rng = 3
            
        escape_count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < wrld.width() and 0 <= ny < wrld.height() and 
                not wrld.wall_at(nx, ny) and 
                not self._is_tile_dangerous(wrld, nx, ny)):
                escape_count += 1
        
        return escape_count

    def _update_epsilon(self):
        # Linear decay
        frac = min(1.0, self.total_steps / float(self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    # ---------- Features ----------
    def _featurize(self, wrld):
        me = self._find_me(wrld)
        if me is None:
            # If somehow not found (dead), return zeros
            C = 6
            H = W = 2 * self.radius + 1
            return torch.zeros((C, H, W), dtype=torch.float32)

        cx, cy = me.x, me.y
        r = self.radius
        H = W = 2 * r + 1

        walls = torch.zeros((H, W), dtype=torch.float32)
        mons  = torch.zeros((H, W), dtype=torch.float32)
        bombs = torch.zeros((H, W), dtype=torch.float32)
        expls = torch.zeros((H, W), dtype=torch.float32)
        exitc = torch.zeros((H, W), dtype=torch.float32)
        mech  = torch.zeros((H, W), dtype=torch.float32)

        def to_patch(x, y):
            return y - (cy - r), x - (cx - r)  # (row, col)

        for px in range(cx - r, cx + r + 1):
            for py in range(cy - r, cy + r + 1):
                pr, pc = to_patch(px, py)
                if px < 0 or py < 0 or px >= wrld.width() or py >= wrld.height():
                    walls[pr, pc] = 1.0
                elif wrld.wall_at(px, py):
                    walls[pr, pc] = 1.0

        # Bombs (with timer info)
        for b in getattr(wrld, "bombs", {}).values():
            bx, by = b.x, b.y
            if abs(bx - cx) <= r and abs(by - cy) <= r:
                pr, pc = to_patch(bx, by)
                if 0 <= pr < H and 0 <= pc < W:
                    # Higher value for bombs about to explode
                    bombs[pr, pc] = max(bombs[pr, pc], 1.0 - (float(b.timer) / 4.0))
        
        # Monsters
        for mlist in getattr(wrld, "monsters", {}).values():
            # Handle both single monsters and lists
            if isinstance(mlist, list):
                for m in mlist:
                    mx, my = m.x, m.y
                    if abs(mx - cx) <= r and abs(my - cy) <= r:
                        pr, pc = to_patch(mx, my)
                        if 0 <= pr < H and 0 <= pc < W:
                            mons[pr, pc] = 1.0
            elif hasattr(mlist, 'x') and hasattr(mlist, 'y'):
                mx, my = mlist.x, mlist.y
                if abs(mx - cx) <= r and abs(my - cy) <= r:
                    pr, pc = to_patch(mx, my)
                    if 0 <= pr < H and 0 <= pc < W:
                        mons[pr, pc] = 1.0
        
        # Explosions
        for e in getattr(wrld, "explosions", {}).values():
            ex, ey = e.x, e.y
            if abs(ex - cx) <= r and abs(ey - cy) <= r:
                pr, pc = to_patch(ex, ey)
                if 0 <= pr < H and 0 <= pc < W:
                    expls[pr, pc] = 1.0

        # Exit
        if wrld.exitcell is not None:
            ex, ey = wrld.exitcell
            if abs(ex - cx) <= r and abs(ey - cy) <= r:
                pr, pc = to_patch(ex, ey)
                if 0 <= pr < H and 0 <= pc < W:
                    exitc[pr, pc] = 1.0

        # Me
        pr, pc = to_patch(cx, cy)
        if 0 <= pr < H and 0 <= pc < W:
            mech[pr, pc] = 1.0

        return torch.stack([walls, mons, bombs, expls, exitc, mech], dim=0)

    def _copy_world(self, wrld):
        try:
            if hasattr(wrld, "copy"):
                return wrld.copy()
        except Exception:
            pass
        return copy.deepcopy(wrld)

    def _find_me(self, wrld):
        try:
            my = wrld.me(self)
            if my is not None and getattr(my, "name", None) == getattr(self, "name", None):
                return my
        except Exception:
            pass
        return None

    def _same_char(self, c):
        try:
            return getattr(c, "name", None) == getattr(self, "name", None)
        except Exception:
            return False

    def _is_tile_dangerous(self, wrld, x, y):
        try:
            rng = wrld.expl_range
        except Exception:
            rng = 3
        if (x, y) in getattr(wrld, "explosions", {}):
            return True
        for b in getattr(wrld, "bombs", {}).values():
            bx, by = b.x, b.y
            if b.timer <= 2:  # imminent
                if x == bx and abs(y - by) <= rng:
                    # Check vertical line
                    step = 1 if y > by else -1
                    yy = by
                    for _ in range(rng + 1):
                        if yy == y:
                            return True
                        if yy < 0 or yy >= wrld.height() or wrld.wall_at(x, yy):
                            break
                        yy += step
                elif y == by and abs(x - bx) <= rng:
                    # Check horizontal line
                    step = 1 if x > bx else -1
                    xx = bx
                    for _ in range(rng + 1):
                        if xx == x:
                            return True
                        if xx < 0 or xx >= wrld.width() or wrld.wall_at(xx, y):
                            break
                        xx += step
        return False

    def save_model(self, path="qchar_model.pt"):
        """Save the model and complete training state"""
        buffer_data = list(self.buffer.buf)
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'target_state_dict': self.target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'replay_buffer': buffer_data,
            'last_bomb_step': self.last_bomb_step,
        }, path)
        print(f"Model saved to {path} (buffer size: {len(buffer_data)})")

    def load_model(self, path="qchar_model.pt"):
        """Load the model and complete training state"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.target.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_steps = checkpoint['total_steps']
            self.epsilon = checkpoint['epsilon']
            
            # Restore replay buffer
            if 'replay_buffer' in checkpoint:
                buffer_data = checkpoint['replay_buffer']
                self.buffer = ReplayBuffer(self.buffer.buf.maxlen)
                for transition in buffer_data:
                    self.buffer.buf.append(transition)
                print(f"Restored replay buffer with {len(buffer_data)} experiences")
            
            # Restore last bomb step
            if 'last_bomb_step' in checkpoint:
                self.last_bomb_step = checkpoint['last_bomb_step']
            
            print(f"Model loaded from {path} (steps: {self.total_steps}, epsilon: {self.epsilon:.3f})")
        except FileNotFoundError:
            print(f"No saved model found at {path}, starting fresh")
        except Exception as e:
            print(f"Error loading model: {e}, starting fresh")
