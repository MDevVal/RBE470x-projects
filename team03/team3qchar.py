import math
import random
from collections import deque

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

        self._last_state = None
        self._last_action = None
        self._alive_last_tick = True

    def do(self, wrld):
        reward, done = self._compute_reward(wrld)

        state = self._featurize(wrld).to(self.device)

        if self._last_state is not None and self._last_action is not None:
            self.buffer.push(self._last_state.detach().cpu(), self._last_action, reward, state.detach().cpu(), done)
            if (self.total_steps % self.train_every) == 0 and len(self.buffer) >= self.batch_size:
                self._optimize()

        a_idx = self._select_action(state)
        name, (dx, dy), bomb = self.ACTIONS[a_idx]

        if bomb:
            self.place_bomb()
        else:
            self.move(dx, dy)

        self._last_state = state
        self._last_action = a_idx
        self.total_steps += 1
        self._update_epsilon()
        if self.total_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.policy.state_dict())

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

    def _select_action(self, state_t):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.policy(state_t.unsqueeze(0))  # (1, A)
            return int(q.argmax(dim=1).item())

    def _update_epsilon(self):
        # Linear decay
        frac = min(1.0, self.total_steps / float(self.eps_decay_steps))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

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

        # Bombs
        for b in getattr(wrld, "bombs", {}).values():
            bx, by = b.x, b.y
            if abs(bx - cx) <= r and abs(by - cy) <= r:
                pr, pc = to_patch(bx, by)
                if 0 <= pr < H and 0 <= pc < W:
                    # normalize timer ~ [0,1], shorter timer -> larger value
                    bombs[pr, pc] = 1.0 / (1.0 + float(b.timer))


        # Explosions
        for e in getattr(wrld, "explosions", {}).values():
            ex, ey = e.x, e.y
            if abs(ex - cx) <= r and abs(ey - cy) <= r:
                pr, pc = to_patch(ex, ey)
                if 0 <= pr < H and 0 <= pc < W:
                    expls[pr, pc] = 1.0 / (1.0 + float(e.timer))

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

        # Stack into (C,H,W)
        return torch.stack([walls, mons, bombs, expls, exitc, mech], dim=0)

    def _compute_reward(self, wrld):
        me = self._find_me(wrld)
        alive = me is not None
        done = False
        reward = 0.0

        # Terminal conditions
        if not alive and self._alive_last_tick:
            reward -= 80.0
            done = True

        # Parse events
        for e in getattr(wrld, "events", []):
            if e.tpe == Event.BOMB_HIT_MONSTER and getattr(e, "character", None) is not None:
                if self._same_char(e.character):
                    reward += 50.0
            elif e.tpe == Event.CHARACTER_KILLED_BY_MONSTER and self._same_char(e.character):
                reward -= 80.0
                done = True
            elif e.tpe == Event.CHARACTER_FOUND_EXIT and self._same_char(e.character):
                reward += 200.0
                done = True
            elif e.tpe == Event.BOMB_HIT_CHARACTER and self._same_char(e.other):
                reward -= 80.0
                done = True

        # Shaping: small time penalty to encourage finishing
        reward -= 1.0

        if alive and self._is_tile_dangerous(wrld, me.x, me.y):
            reward -= 5.0

        self._alive_last_tick = alive

        if alive and getattr(wrld, "exitcell", None) is not None and me is not None:
            ex, ey = wrld.exitcell
            prev_dist = getattr(self, "_prev_exit_dist", None)
            curr_dist = abs(me.x - ex) + abs(me.y - ey)
            if prev_dist is not None:
                reward += 0.3 * (prev_dist - curr_dist)
            self._prev_exit_dist = curr_dist
        else:
            self._prev_exit_dist = None
        
        return reward, done

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
                if x == bx:
                    step = 1 if y > by else -1
                    yy = by
                    for _ in range(rng):
                        yy += step
                        if yy < 0 or yy >= wrld.height(): break
                        if wrld.wall_at(x, yy): break
                        if (x, yy) == (x, y): return True
                elif y == by:
                    step = 1 if x > bx else -1
                    xx = bx
                    for _ in range(rng):
                        xx += step
                        if xx < 0 or xx >= wrld.width(): break
                        if wrld.wall_at(xx, y): break
                        if (xx, y) == (x, y): return True
        return False
