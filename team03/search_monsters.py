from itertools import product
from typing import Iterable, List, Tuple

Coord = Tuple[int, int]

DIRS8 = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]

def inb(grid, x: int, y: int) -> bool:
    return 0 <= x < len(grid) and 0 <= y < len(grid[x])

def cheb(a: Coord, b: Coord) -> int:
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def free_neighbors(grid, x: int, y: int) -> Iterable[Coord]:
    for dx, dy in DIRS8 + [(0,0)]:  # allow wait
        nx, ny = x + dx, y + dy
        if inb(grid, nx, ny) and not grid[nx][ny]:
            yield (nx, ny)

def predict_one_step_toward(grid, px: int, py: int, mx: int, my: int) -> Coord:
    best = (cheb((px,py),(mx,my)), mx, my)
    for dx, dy in DIRS8:
        nx, ny = mx + dx, my + dy
        if inb(grid, nx, ny) and not grid[nx][ny]:
            d = cheb((px,py),(nx,ny))
            if d < best[0]:
                best = (d, nx, ny)
    return best[1], best[2]

def monsters_next_positions(grid, px: int, py: int, monsters: Iterable[Coord], cap_per_mon: int = 3):
    """
    Worst-case branching: each monster takes any best step(s) toward the player.
    Capped per monster to control branching blow-up.
    """
    per_mon_opts: List[List[Coord]] = []
    for mx, my in monsters:
        best_d = cheb((px,py),(mx,my))
        opts = set()
        stay = (mx, my)
        for dx, dy in DIRS8 + [(0,0)]:
            nx, ny = mx + dx, my + dy
            if inb(grid, nx, ny) and not grid[nx][ny]:
                d = cheb((px,py),(nx,ny))
                if d < best_d:
                    best_d = d; opts = {(nx, ny)}
                elif d == best_d:
                    opts.add((nx, ny))
        if not opts:
            opts = {stay}
        per_mon_opts.append(list(opts)[:cap_per_mon])
    for combo in product(*per_mon_opts):
        yield tuple(combo)

