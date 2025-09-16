# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math, heapq, time

class BenniChar(CharacterEntity):
    reset = False
    inverted = False

    DIRS = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

    def inb(self, grid, x, y):
        return 0 <= x < len(grid) and 0 <= y < len(grid[x])

    def predict_one_step_toward(self, px, py, mx, my, grid):
        best = (max(abs(px - mx), abs(py - my)), mx, my)
        for dx, dy in self.DIRS:
            nx, ny = mx + dx, my + dy
            if self.inb(grid, nx, ny) and not grid[nx][ny]:
                d = max(abs(px - nx), abs(py - ny))
                if d < best[0]:
                    best = (d, nx, ny)
        return best[1], best[2]

    def a_star(self, grid, start, goal, monsters):
        sx, sy = start
        gx, gy = goal

        forbidden = set(monsters)
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

    def next_move(self, wrld):
        W = len(wrld.grid)
        H = len(wrld.grid[0]) if W > 0 else 0
        goal = (W - 1, H - 1)
        start = (self.x, self.y)
        monsters = [(m.x, m.y) for group in wrld.monsters.values() for m in group]
        path = self.a_star(wrld.grid, start, goal, monsters)
        if not path or len(path) < 2:
            return 0, 0
        nx, ny = path[1]
        return nx - self.x, ny - self.y

    def do(self, wrld):
        dx, dy = self.next_move(wrld)

        bomb = False
        col = wrld.grid[self.x]
        print(wrld.grid)
        lowerPos = col[self.y + 1] if (self.y + 1) < len(col) else True

        closestEnemyDistance = min(
            (math.dist((self.x, self.y), (m.x, m.y)) for g in wrld.monsters.values() for m in g),
            default=0
        )
        print(closestEnemyDistance)

        if lowerPos == False and self.inb(wrld.grid, self.x, self.y + 1) and not wrld.grid[self.x][self.y + 1]:
            dy = 1

        time.sleep(0.02)
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

