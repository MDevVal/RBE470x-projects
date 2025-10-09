# find bomberman engine
import sys
sys.path.insert(0, '../bomberman')

# core bits
from entity import CharacterEntity
from colorama import Fore, Back


# usual libs
import numpy as np
from priority_queue import PriorityQueue
from enum import Enum
from heapq import heappush, heappop

import copy
import random

class QChar(CharacterEntity):
    # feature order matters w/ weights
    features = [
        "EXIT_DISTANCE",
        "IS_ALIVE",
        "NUMBER_OF_WALLS",
        "BOMB_PLACED",
        "IN_BOMB_RANGE",
        "DISTANCE_FROM_BOMB",
        "DISTANCE_FROM_MONSTER",
    ]
    num_features = len(features)
    weights = []  # load or make rand

    # actions (diag on) -> [[dx,dy], bomb_flag]
    diagactions = [
        [[0, 1], 0], [[0, -1], 0], [[-1, 0], 0], [[1, 0], 0],
        [[0, 0], 1],
        [[1, 1], 0], [[-1, 1], 0], [[1, -1], 0], [[-1, -1], 0]
    ]
    actions = diagactions  # keep diag

    # one time caches
    init_flag = False
    exit_wavefront = None      # wave frm exit no walls
    to_goal_wave = None        # wave frm exit w walls
    world_size = None          # big num for norm
    goal = None                # exit (x,y)
    isBomb = False
    maxTime = 0

    # per turn stuff
    position = None            # curr (x,y)
    reachable = None           # not used rn

    def __init__(self, name, color, x, y):
        super().__init__(name, color, x, y)
        # load weights or make new ones
        try:
            self.weights = np.loadtxt("weights.csv", delimiter=",", dtype=float).tolist()
            print("Inital Weighs: ", self.weights)
            if len(self.weights) != self.num_features:
                raise ValueError("bad len")
            if np.isnan(self.weights).any():
                raise ValueError("nan in weights")
        except Exception:
            print("Picking random weights")
            self.weights = [random.random() for _ in range(self.num_features)]
            try:
                np.savetxt("weights.csv", self.weights)
            except Exception:
                pass

    # --- main loop ---
    def do(self, wrld):
        # lazy init first tick
        if not self.init_flag:
            self.goal = self.findExit(wrld)
            self.exit_wavefront = self.generateWavefront(wrld, self.goal, True)
            self.to_goal_wave = self.generateWavefront(wrld, self.goal, False)
            self.world_size = np.max(self.exit_wavefront)
            self.maxWalls = self.findWalls(wrld)
            self.init_flag = True
            self.isBomb = False
            self.maxTime = wrld.time

        # if expl happened refresh nowalls wave
        if len(wrld.explosions) > 0:
            self.exit_wavefront = self.generateWavefront(wrld, self.goal, True)

        # curr pos + defaults
        self.position = (self.x, self.y)
        dx, dy, bomb = 0, 0, False

        # try fast path to exit if safe ish
        path = self.astar(wrld, self.position, self.goal, withExplosions=True, diag=True)
        monsters = self.findMonsters(wrld)
        print("Path: ", path)  # debug print i keep

        # no mons -> just go next step
        if path and not monsters:
            nx, ny = path[1]
            self.move(nx - self.x, ny - self.y)
            return

        # mons exist but we r closer to exit by a bit -> still go
        if path:
            dist_to_goal = len(path)
            m_best = float('inf')
            for m in monsters:
                d = len(self.astar(wrld, m, self.goal))
                if d > 0:
                    m_best = min(m_best, d)
            if dist_to_goal < (m_best - 2):
                nx, ny = path[1]
                self.move(nx - self.x, ny - self.y)
                return

        # else use q pick
        pick = self.argMax([(wrld, a) for a in self.actions], self.getQValue)
        if pick is None:
            action = self.actions[random.randint(0, len(self.actions) - 1)]
        else:
            action = pick[1]

        (dx, dy), bomb = action[0], action[1]

        delta = self.getDelta(wrld, action)
        self.updateWeights(self.getFeatureValues(wrld), delta)
        try:
            np.savetxt("weights.csv", self.weights)
        except Exception:
            pass
        
        self.move(dx, dy)
        if bomb:
            self.place_bomb()

    # --- q bits ---
    def getDelta(self, wrld, action_taken):
        # td(0) target
        gamma = 0.9
        next_wrld = self.result(wrld, action_taken)

        # greedy next (fallback rand)
        next_pick = self.argMax([(next_wrld, a) for a in self.actions], self.getQValue)
        next_action = next_pick[1] if next_pick is not None else self.actions[random.randint(0, len(self.actions) - 1)]

        r = self.getReward(next_wrld)
        return r + gamma * self.getQValue(next_wrld, next_action) - self.getQValue(wrld, action_taken)

    def updateWeights(self, feature_values, delta):
        # small step
        alpha = 0.005
        for i in range(self.num_features):
            self.weights[i] += alpha * -delta * feature_values[i]

    def getQValue(self, wrld, action):
        # sim one step then score
        nxt = self.result(wrld, action)
        if nxt is None:
            return -10000  # dead branch
        # if char gone in nxt, rip
        if self.findChar(nxt.next()[0]) == (-1, -1):
            return -10000
        return float(np.dot(self.weights, self.getFeatureValues(nxt)))

    # --- features + reward ---
    def getFeatureValues(self, wrld):
        # keep same order as self.features
        return [self.featureValue(wrld, name) for name in self.features]

    def featureValue(self, wrld, name):
        pos = self.findChar(wrld)  # (x,y)

        if name == "EXIT_DISTANCE":
            dist = self.exit_wavefront[pos[0]][pos[1]]
            return (self.world_size - dist) / self.world_size

        if name == "BOMB_PLACED":
            return 1.0 if self.findBomb(wrld) else 0.0

        if name == "IS_ALIVE":
            return 0.0 if self.findChar(wrld) == (-1, -1) else 1.0

        if name == "NUMBER_OF_WALLS":
            count = self.findWalls(wrld)
            return (self.maxWalls - count) / self.maxWalls

        if name == "IN_BOMB_RANGE":
            b = self.findBomb(wrld)
            if b:
                t = self.checkTimeToExplode(wrld, b, pos)
                if t >= 1:
                    return 1 - 1 / t
            return 1.0

        if name == "DISTANCE_FROM_BOMB":
            b = self.findBomb(wrld)
            ex = self.findExplosion(wrld)
            if ex:
                dist = float('inf')
                for e in ex:
                    d = len(self.astar(wrld, pos, (e[0], e[1])))
                    if d > 0:
                        dist = min(dist, d)
                if dist != float('inf'):
                    return 1 - 1 / (dist + 0.1)
            if b:
                d = len(self.astar(wrld, pos, b))
                return 1 - 1 / (d + 0.1)
            return 1.0

        if name == "DISTANCE_FROM_MONSTER":
            mons = self.findMonsters(wrld)
            if mons:
                dist = float('inf')
                for m in mons:
                    d = len(self.astar(wrld, pos, m, throughWalls=False))
                    if d > 0:
                        dist = min(dist, d)
                return 1 - 1 / dist
            return 0.0

        return 0.0

    def getReward(self, wrld):
        # shaping: stay alive + closer exit + bombs use + away from mons
        cx, cy = self.findChar(wrld)
        if (cx, cy) == (-1, -1):
            return -10000

        bombPoints = 0
        b = self.findBomb(wrld)
        if b:
            bombPoints = 500
            d = len(self.astar(wrld, (cx, cy), b))
            if d > 0:
                bombPoints = 500 + 2 * d

        dist_exit = self.exit_wavefront[cx][cy]

        # blast = not good
        if wrld.explosion_at(cx, cy):
            return -10000
        if b and self.checkTimeToExplode(wrld, b, (cx, cy)) == 1:
            return -10000

        # farther from mons good
        monstPoints = 0
        mons = self.findMonsters(wrld)
        if mons:
            dist = float('inf')
            for m in mons:
                d = len(self.astar(wrld, (cx, cy), m, throughWalls=False, withExplosions=True))
                if d > 0:
                    dist = min(dist, d)
            if dist != float('inf'):
                monstPoints = 6 * dist

        # base score is -time so add extras
        return wrld.scores["me"] - dist_exit + bombPoints + monstPoints

    #  graph helpers
    def reachableCells(self, wrld, point):
        # calc cells we can reach from here
        try:
            nw = wrld.from_world(wrld)
            nx, ny = nw.me(self).x, nw.me(self).y
            wave = self.generateWavefront(nw, (nx, ny))
            out = []
            for r in range(len(wave)):
                for c in range(len(wave[0])):
                    if wave[r][c] < float('inf'):
                        out.append((c, r))
            return out
        except Exception as e:
            print("Error in reachableCells:", e)
            return []

    def astar(self, wrld, start, goal, throughWalls=False, withExplosions=False, diag=False):
        # a* on grid w/ flags
        path = []
        found = False

        pq = PriorityQueue()
        pq.put((0, tuple(start), None), 0)
        explored = {}

        while not found and not pq.empty():
            g, exploring, parent = pq.get()
            explored[exploring] = (g, exploring, parent)

            if exploring == tuple(goal):
                found = True
                break

            for neighbor in self.getNeighbors(
                wrld, exploring,
                throughWalls=throughWalls,
                withExplosions=withExplosions,
                diag=diag,
                turns=1
            ):
                if neighbor not in explored:
                    gfactor = 1
                    if throughWalls and wrld.wall_at(neighbor[0], neighbor[1]):
                        gfactor += wrld.bomb_time
                    f = g + gfactor + self.manhattan(neighbor, goal)
                    pq.put((g + gfactor, neighbor, exploring), f)

        if found:
            path = self.reconstructPath(explored, tuple(start), tuple(goal))
        return path

    def getNeighbors(self, wrld, cell, withBomb=False, withMonster=False,
                     throughWalls=False, withExplosions=False, diag=False, turns=1):
        # neighbors that r walkable based on flags
        x, y = cell
        rows, cols = wrld.height(), wrld.width()
        if x < 0 or x >= cols or y < 0 or y >= rows:
            return []
        if wrld.wall_at(x, y) == 1:
            return []

        dirs = self.diagactions if diag else self.actions
        out = []
        for act in dirs:
            dx, dy = act[0]
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                if not withExplosions or not wrld.explosion_at(nx, ny):
                    if throughWalls or wrld.wall_at(nx, ny) == 0:
                        if not withBomb or 0 > self.checkTimeToExplode(wrld, self.findBomb(wrld), (nx, ny), turns):
                            if not withMonster or not wrld.monsters_at(nx, ny):
                                out.append((nx, ny))
        return out

    def reconstructPath(self, explored: dict, start: tuple[int, int], goal: tuple[int, int]) -> list[list[int, int]]:
        # backtrack path from explored
        cur = goal
        path = []
        while cur != start:
            g, here, parent = explored[cur]
            path = [list(cur)] + path
            cur = parent
            if cur is None:  # safety
                return []
        return [list(start)] + path

    def generateWavefront(self, wrld, start, throughWalls=False):
        # dijkstra style wave. walls cost bomb_time if allowed
        rows, cols = wrld.height(), wrld.width()
        wave = [[float('inf') for _ in range(rows)] for _ in range(cols)]
        wave[start[0]][start[1]] = 0

        q = PriorityQueue()
        q.put((0, (start[0], start[1])), 0)

        acts = self.actions
        while not q.empty():
            cost, (cx, cy) = q.get()
            wave[cx][cy] = cost

            for act in acts:
                dx, dy = act[0]
                nx, ny = cx + dx, cy + dy
                if 0 <= ny < rows and 0 <= nx < cols:
                    if wrld.wall_at(nx, ny) == 0:
                        nc = cost + 1
                        if nc < wave[nx][ny]:
                            q.put((nc, (nx, ny)), nc)
                    elif throughWalls:
                        nc = cost + 1 + wrld.bomb_time
                        if nc < wave[nx][ny]:
                            q.put((nc, (nx, ny)), nc)
        return wave

    # --- lil helpers ---
    @staticmethod
    def manhattan(p1, p2):
        return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

    @staticmethod
    def openDist(p1, p2):
        return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))

    def findExit(self, wrld):
        ex, ey = 0, 0
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.exit_at(c, r):
                    ey, ex = r, c
        return ex, ey

    def findChar(self, wrld):
        y, x = -1, -1
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.characters_at(c, r):
                    y, x = r, c
        return x, y

    def findMonsters(self, wrld):
        mons = []
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.monsters_at(c, r):
                    mons.append((c, r))
        return mons

    def findExplosion(self, wrld):
        ex = []
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.explosion_at(c, r):
                    ex.append((c, r))
        return ex

    def findBomb(self, wrld):
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.bomb_at(c, r):
                    return (c, r)
        return None

    def checkTimeToExplode(self, wrld, bomb, tile):
        # return timer if tile in bomb line else -1, 1 if already exploding
        if bomb is None:
            return -1
        b = wrld.bomb_at(bomb[0], bomb[1])
        if b and b.timer:
            same_col = (abs(tile[0] - bomb[0]) <= 4 and tile[1] == bomb[1])
            same_row = (abs(tile[1] - bomb[1]) <= 4 and tile[0] == bomb[0])
            if same_col or same_row:
                return b.timer
        if wrld.explosion_at(tile[0], tile[1]):
            return 1
        return -1

    def findWalls(self, wrld):
        cnt = 0
        for r in range(wrld.height()):
            for c in range(wrld.width()):
                if wrld.wall_at(c, r):
                    cnt += 1
        return cnt

    def result(self, wrld, action):
        # simulate one step and return nxt world or None
        new_world = wrld.from_world(wrld)
        try:
            (mx, my), bomb = action[0], action[1]
            if bomb == 1:
                new_world.me(self).place_bomb()
            new_world.me(self).move(mx, my)
            nxt, _ = new_world.next()
            return nxt
        except Exception as e:
            # try stepping anyway
            try:
                nxt, _ = new_world.next()
                return nxt
            except Exception:
                return None

    @staticmethod
    def argMax(args, util_function):
        # pick arg w/ max util
        best = float('-inf')
        arg_best = None
        for a in args:
            val = util_function(*a)
            if val > best:
                best = val
                arg_best = a
        return arg_best
