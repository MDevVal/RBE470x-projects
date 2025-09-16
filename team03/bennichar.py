import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
import math, heapq, collections

class BenniChar(CharacterEntity):
    reset = False
    inverted = False
    DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    ORTHO_DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
    BOMB_TIME = 10
    EXPL_DURATION = 2
    EXPL_RANGE = 4
    plan_path = None
    plan_step = 0
    bomb_armed = False
    bomb_countdown = 0
    planned_blast = set()

    def inb(self, grid, x, y):
        return 0 <= x < len(grid) and 0 <= y < len(grid[x])

    def cheb(self, a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def get_monsters(self, wrld):
        return [(m.x, m.y) for group in wrld.monsters.values() for m in group]

    def predict_one_step_toward(self, px, py, mx, my, grid):
        best = (self.cheb((px,py),(mx,my)), mx, my)
        for dx, dy in self.DIRS:
            nx, ny = mx + dx, my + dy
            if self.inb(grid, nx, ny) and not grid[nx][ny]:
                d = self.cheb((px,py),(nx,ny))
                if d < best[0]:
                    best = (d, nx, ny)
        return best[1], best[2]

    def a_star(self, grid, start, goal, monsters, avoid_pred_steps=1):
        sx, sy = start
        gx, gy = goal
        forbidden = set(monsters)
        if avoid_pred_steps >= 1:
            for mx, my in monsters:
                nx, ny = self.predict_one_step_toward(sx, sy, mx, my, grid)
                forbidden.add((nx, ny))
        def h(x, y):
            return max(abs(x - gx), abs(y - gy))
        g = {start: 0.0}
        pq = [(h(sx, sy), 0.0, sx, sy)]
        parent = {}
        while pq:
            f, gc, x, y = heapq.heappop(pq)
            if (x, y) == (gx, gy):
                path = []
                cur = (x, y)
                while cur in parent:
                    path.append(cur)
                    cur = parent[cur]
                path.append(start)
                path.reverse()
                return path
            for dx, dy in self.DIRS:
                nx, ny = x + dx, y + dy
                if not self.inb(grid, nx, ny) or grid[nx][ny]:
                    continue
                if (nx, ny) in forbidden:
                    continue
                ng = gc + 1.0
                if ng < g.get((nx, ny), 1e18):
                    g[(nx, ny)] = ng
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))
        return None

    def clear_line(self, grid, x, y, dx, dy, rng):
        cells = []
        cx, cy = x, y
        for _ in range(rng):
            cx += dx; cy += dy
            if not self.inb(grid, cx, cy):
                break
            if grid[cx][cy]:
                break
            cells.append((cx, cy))
        return cells

    def blast_cross(self, grid, bx, by, rng):
        cells = set()
        cells.add((bx, by))
        for dx, dy in self.ORTHO_DIRS:
            for c in self.clear_line(grid, bx, by, dx, dy, rng):
                cells.add(c)
        return cells

    def aligned_no_wall(self, grid, a, b):
        (ax, ay), (bx, by) = a, b
        if ax == bx:
            dy = 1 if by > ay else -1
            for cy in range(ay + dy, by, dy):
                if grid[ax][cy]:
                    return False
            return True
        if ay == by:
            dx = 1 if bx > ax else -1
            for cx in range(ax + dx, bx, dx):
                if grid[cx][ay]:
                    return False
            return True
        return False

    def simulate_monster_until(self, grid, mpos, target, steps):
        mx, my = mpos
        path = [mpos]
        for _ in range(steps):
            mx, my = self.predict_one_step_toward(target[0], target[1], mx, my, grid)
            path.append((mx, my))
        return path

    def any_monster_in_blast_at_time(self, grid, monsters, bomb_pos, target_player_pos, t_at_blast):
        bx, by = bomb_pos
        blast = self.blast_cross(grid, bx, by, self.EXPL_RANGE)
        for m in monsters:
            mpath = self.simulate_monster_until(grid, m, target_player_pos, t_at_blast)
            if mpath[-1] in blast:
                return True
        return False

    def find_time_safe_escape(self, grid, start, forbidden_now, blast_cells_at_blast, bomb_time, expl_duration, monsters):
        monster_forbid_per_t = [set() for _ in range(bomb_time + 1)]
        monster_forbid_per_t[0] = set(forbidden_now)
        for t in range(1, bomb_time + 1):
            forb = set()
            for m in monsters:
                mx, my = m
                for _ in range(t):
                    mx, my = self.predict_one_step_toward(start[0], start[1], mx, my, grid)
                forb.add((mx, my))
            monster_forbid_per_t[t] = forb
        Q = collections.deque()
        Q.append((start[0], start[1], 0))
        parent = {}
        seen = {(start[0], start[1], 0)}
        while Q:
            x, y, t = Q.popleft()
            if t == bomb_time:
                safe = True
                for _dt in range(expl_duration):
                    if (x, y) in blast_cells_at_blast:
                        safe = False
                        break
                if safe:
                    path = []
                    cur = (x, y, t)
                    while cur in parent:
                        path.append((cur[0], cur[1]))
                        cur = parent[cur]
                    path.append((start[0], start[1]))
                    path.reverse()
                    return path
            for dx, dy in self.DIRS:
                nx, ny = x + dx, y + dy
                nt = t + 1
                if nt > bomb_time:
                    continue
                if not self.inb(grid, nx, ny) or grid[nx][ny]:
                    continue
                if (nx, ny) in monster_forbid_per_t[nt]:
                    continue
                state = (nx, ny, nt)
                if state in seen:
                    continue
                seen.add(state)
                parent[state] = (x, y, t)
                Q.append(state)
        return None

    def find_exit_or_corner(self, wrld):
        if hasattr(wrld, 'exitcell') and wrld.exitcell is not None:
            return wrld.exitcell
        W = len(wrld.grid)
        H = len(wrld.grid[0]) if W > 0 else 0
        gx, gy = W - 2, H - 2
        for rx in range(W - 1, -1, -1):
            for ry in range(H - 1, -1, -1):
                if self.inb(wrld.grid, rx, ry) and not wrld.grid[rx][ry]:
                    return (rx, ry)
        return (self.x, self.y)

    def next_move(self, wrld):
        grid = wrld.grid
        me = (self.x, self.y)
        monsters = self.get_monsters(wrld)
        if self.bomb_armed and self.plan_path:
            if self.plan_step + 1 < len(self.plan_path):
                nxt = self.plan_path[self.plan_step + 1]
                self.plan_step += 1
                return nxt[0] - self.x, nxt[1] - self.y
            else:
                return 0, 0
        goal = self.find_exit_or_corner(wrld)
        path = self.a_star(grid, me, goal, monsters, avoid_pred_steps=1)
        should_bomb = False
        bomb_pos = me
        aligned_threat = None
        for m in monsters:
            if self.aligned_no_wall(grid, me, m):
                man = abs(m[0]-me[0]) + abs(m[1]-me[1])
                if man <= self.EXPL_RANGE:
                    aligned_threat = m
                    break
        if aligned_threat is not None:
            can_hit = self.any_monster_in_blast_at_time(
                grid, monsters, bomb_pos, me, self.BOMB_TIME
            )
            if can_hit:
                blast_cells = self.blast_cross(grid, bomb_pos[0], bomb_pos[1], self.EXPL_RANGE)
                forbidden_now = set(monsters)
                for mx, my in monsters:
                    nx, ny = self.predict_one_step_toward(me[0], me[1], mx, my, grid)
                    forbidden_now.add((nx, ny))
                plan = self.find_time_safe_escape(
                    grid, me, forbidden_now, blast_cells,
                    self.BOMB_TIME, self.EXPL_DURATION, monsters
                )
                if plan:
                    should_bomb = True
                    self.plan_path = plan
                    self.plan_step = 0
                    self.bomb_armed = True
                    self.bomb_countdown = self.BOMB_TIME
                    self.planned_blast = blast_cells
        if should_bomb and self.plan_path and len(self.plan_path) >= 2:
            nx, ny = self.plan_path[1]
            return nx - self.x, ny - self.y
        if not path or len(path) < 2:
            return 0, 0
        nx, ny = path[1]
        return nx - self.x, ny - self.y

    def do(self, wrld):
        if self.bomb_armed:
            self.bomb_countdown -= 1
            if self.bomb_countdown <= -(self.EXPL_DURATION):
                self.bomb_armed = False
                self.plan_path = None
                self.plan_step = 0
                self.planned_blast = set()
        dx, dy = self.next_move(wrld)
        bomb = False
        if self.bomb_armed and self.plan_step == 0 and self.plan_path and self.plan_path[0] == (self.x, self.y):
            bomb = True
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

