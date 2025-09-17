import sys
sys.path.insert(0, '../bomberman')
from entity import CharacterEntity
import math, heapq, collections

class Team3Char(CharacterEntity):
    # Fallbacks if the world doesn't expose these
    BOMB_TIME = 10
    EXPL_DURATION = 2
    EXPL_RANGE = 4

    DIRS_8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    DIRS_4 = [(1,0),(-1,0),(0,1),(0,-1)]

    # Planning state
    plan_path = None
    plan_step = 0
    bomb_armed = False
    bomb_countdown = 0
    planned_blast = set()


    # Helpers
    def _inb(self, grid, x, y):
        return 0 <= x < len(grid) and 0 <= y < len(grid[0])

    def _cheb(self, a, b):
        return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

    def _monsters(self, wrld):
        return [(m.x, m.y) for group in wrld.monsters.values() for m in group]

    def _world_params(self, wrld):
        bt = getattr(wrld, 'bomb_time', None)
        ed = getattr(wrld, 'expl_duration', None)
        er = getattr(wrld, 'expl_range', None)
        world = getattr(wrld, 'world', None)
        if (bt is None or ed is None or er is None) and world is not None:
            bt = bt if bt is not None else getattr(world, 'bomb_time', None)
            ed = ed if ed is not None else getattr(world, 'expl_duration', None)
            er = er if er is not None else getattr(world, 'expl_range', None)
        return (bt or self.BOMB_TIME, ed or self.EXPL_DURATION, er or self.EXPL_RANGE)


    # Threat zone estimation
    def _monster_reachable_next(self, grid, mpos):
        """all squares a monster could occupy next turn (8-dir )"""
        out = set()
        for dx, dy in self.DIRS_8 + [(0,0)]:
            nx, ny = mpos[0] + dx, mpos[1] + dy
            if self._inb(grid, nx, ny) and not grid[nx][ny]:
                out.add((nx, ny))
        return out

    def _compute_threat_zones(self, grid, monsters):
        """all monsters next step reach (t=1) and a conservative t=2 expansion"""
        t1 = set()
        for m in monsters:
            t1 |= self._monster_reachable_next(grid, m)

        # conservative t=2: expand from all t1 cells
        t2 = set(t1)
        for c in list(t1):
            for dx, dy in self.DIRS_8 + [(0,0)]:
                nx, ny = c[0] + dx, c[1] + dy
                if self._inb(grid, nx, ny) and not grid[nx][ny]:
                    t2.add((nx, ny))
        return t1, t2

    # Soft cost A*
    def _danger_cost(self, grid, monsters, threat1, threat2, x, y):
        # hard avoid
        if (x, y) in monsters:
            return float('inf')

        cost = 1.0

        # distance-based soft cost to stay away from mons
        min_cheb = min((self._cheb((x, y), m) for m in monsters), default=99)
        if min_cheb <= 1:
            return float('inf')          # never step adjacent if possible to avoide
        elif min_cheb == 2:
            cost += 8.0
        elif min_cheb == 3:
            cost += 4.5
        elif min_cheb == 4:
            cost += 2.0

        # Strong bias away from threat zones
        if (x, y) in threat1:
            cost += 50.0
        elif (x, y) in threat2:
            cost += 15.0

        return cost

    def _astar_soft(self, grid, start, goal, monsters, threat1, threat2):
        if start == goal:
            return [start]

        W, H = len(grid), len(grid[0])
        def inb(x, y): return 0 <= x < W and 0 <= y < H
        def h(x, y):  return self._cheb((x, y), goal)

        g = {start: 0.0}
        parent = {}
        pq = [(h(*start), 0.0, start[0], start[1])]
        done = set()

        while pq:
            f, gc, x, y = heapq.heappop(pq)
            if (x, y) in done:
                continue
            done.add((x, y))

            if (x, y) == goal:
                path = []
                cur = (x, y)
                while cur in parent:
                    path.append(cur)
                    cur = parent[cur]
                path.append(start)
                path.reverse()
                return path

            for dx, dy in self.DIRS_8:
                nx, ny = x + dx, y + dy
                if not inb(nx, ny) or grid[nx][ny]:
                    continue
                step_cost = self._danger_cost(grid, monsters, threat1, threat2, nx, ny)
                if step_cost == float('inf'):
                    continue
                ng = gc + step_cost
                if ng < g.get((nx, ny), float('inf')):
                    g[(nx, ny)] = ng
                    parent[(nx, ny)] = (x, y)
                    heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))
        return None

    # evasive manuvers
    def _evasion_step(self, wrld, monsters, threat1, threat2, goal):
        """pick safest local  move still roughly toward goal"""
        best = (-1e18, 0, 0)
        # try and precalculate the distance to threat1 for scoring
        def dist_to_set(p, S):
            if not S: return 99
            return min(self._cheb(p, q) for q in S)

        for dx, dy in self.DIRS_8:
            nx, ny = self.x + dx, self.y + dy
            if not self._inb(wrld.grid, nx, ny) or wrld.grid[nx][ny]:
                continue
            # dont stand right next to current monster positions
            min_to_mon = min((self._cheb((nx, ny), m) for m in monsters), default=99)
            if min_to_mon <= 1:
                continue

            # score: (safety from t1/t2) - (tiny pull to goal)
            s1 = dist_to_set((nx, ny), threat1)
            s2 = dist_to_set((nx, ny), threat2)
            toward_exit = -0.4 * self._cheb((nx, ny), goal)  # small bias to still progress so we dont get stuck in that back and forth loop again on v2 or v3
            # extra bonus for staying out of threat entirely
            bonus = 0.0
            if (nx, ny) not in threat1: bonus += 5.0
            if (nx, ny) not in threat2: bonus += 2.0

            score = 6.0 * s1 + 2.0 * s2 + toward_exit + bonus
            if score > best[0]:
                best = (score, dx, dy)

        #if nothing passes filters reverse to any open square not in threat1
        if best[0] <= -1e17:
            fallback = None
            for dx, dy in self.DIRS_8:
                nx, ny = self.x + dx, self.y + dy
                if self._inb(wrld.grid, nx, ny) and not wrld.grid[nx][ny]:
                    if (nx, ny) not in threat1:
                        fallback = (dx, dy)
                        break
            if fallback:
                return fallback
            return (0, 0)
        return (best[1], best[2])

    #goal
    def _find_exit_or_fallback(self, wrld):
        if hasattr(wrld, 'exitcell') and wrld.exitcell:
            return wrld.exitcell
        W, H = len(wrld.grid), len(wrld.grid[0])
        for rx in range(W-1, -1, -1):
            for ry in range(H-1, -1, -1):
                if not wrld.grid[rx][ry]:
                    return (rx, ry)
        return (self.x, self.y)

    def next_move(self, wrld):
        grid = wrld.grid
        me = (self.x, self.y)
        monsters = set(self._monsters(wrld))
        bt, ed, rng = self._world_params(wrld)

        #follow bomb escape plan first
        if self.bomb_armed and self.plan_path:
            if self.plan_step + 1 < len(self.plan_path):
                nxt = self.plan_path[self.plan_step + 1]
                self.plan_step += 1
                return (nxt[0] - self.x, nxt[1] - self.y)
            return (0, 0)

        goal = self._find_exit_or_fallback(wrld)
        threat1, threat2 = self._compute_threat_zones(grid, monsters)

        # safest path to exit with soft costs + threat penalties
        path = self._astar_soft(grid, me, goal, monsters, threat1, threat2)

        #if we have a path but the next step is risky try and sidestep to avoid
        if path and len(path) >= 2:
            nx, ny = path[1]
            next_is_adjacent = min((self._cheb((nx, ny), m) for m in monsters), default=99) <= 1
            if (nx, ny) in threat1 or next_is_adjacent or me in threat1:
                return self._evasion_step(wrld, monsters, threat1, threat2, goal)
            return (nx - self.x, ny - self.y)

        #no path found so it will try and evade locally
        return self._evasion_step(wrld, monsters, threat1, threat2, goal)

    def do(self, wrld):
        #bomb bookkeeping kept conservative   (its rarely used in v2 and v4)
        if self.bomb_armed:
            self.bomb_countdown -= 1
            if self.bomb_countdown <= -(self.EXPL_DURATION):
                self.bomb_armed = False
                self.plan_path = None
                self.plan_step = 0
                self.planned_blast = set()

        dx, dy = self.next_move(wrld)

        #place bomb only if we previously armed
        bomb = False
        if self.bomb_armed and self.plan_step == 0 and self.plan_path and self.plan_path[0] == (self.x, self.y):
            bomb = True

        self.move(dx, dy)
        if bomb:
            self.place_bomb()
