from typing import Dict, List, Tuple, Set

Coord = Tuple[int, int]
Bomb = Dict[str, int]  # keys: x, y, t (real fuse), te (effective fuse), owner: 0/1
Flame = Dict[str, object]  # {'tiles': set[(x,y)], 't': int}

def inb(grid, x: int, y: int) -> bool:
    return 0 <= x < len(grid) and 0 <= y < len(grid[x])

def blast_tiles_for(grid, bx: int, by: int, expl_range: int) -> Set[Coord]:
    """Bomberman cross: up to expl_range tiles in 4 cardinals, blocked by walls."""
    tiles: Set[Coord] = {(bx, by)}
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        x, y = bx, by
        for _ in range(expl_range):
            x += dx; y += dy
            if not inb(grid, x, y): break
            tiles.add((x, y))
            if grid[x][y]:  # wall blocks propagation
                break
    return tiles

def flame_union(flames: List[Flame]) -> Set[Coord]:
    out: Set[Coord] = set()
    for f in flames:
        out |= f['tiles']
    return out

def in_flames(x: int, y: int, flames: List[Flame]) -> bool:
    return any((x, y) in f['tiles'] for f in flames)

def tick_bombs_and_flames(
    grid,
    bombs: List[Bomb],
    flames: List[Flame],
    expl_range: int,
    expl_duration: int
) -> Tuple[List[Bomb], List[Flame], Set[Coord]]:
    """
    Decrement bombs/flames 1 tick.
    - Chain reaction: a bomb inside flame explodes now.
    - Explosion happens when effective fuse (te) reaches 0.
    Returns (bombs_after, flames_after, new_flame_tiles)
    """
    flame_tiles = flame_union(flames)

    to_explode: List[Bomb] = []
    ticking: List[Bomb] = []

    for b in bombs:
        # chain reaction or due
        if (b['x'], b['y']) in flame_tiles or b['te'] <= 1:
            to_explode.append({**b, 't': max(b['t']-1, 0), 'te': 0})
        else:
            ticking.append({**b, 't': max(b['t']-1, 0), 'te': max(b['te']-1, 0)})

    # advance flames
    new_flames: List[Flame] = []
    for f in flames:
        nt = f['t'] - 1
        if nt > 0:
            new_flames.append({'tiles': f['tiles'], 't': nt})

    # new explosions
    new_tiles: Set[Coord] = set()
    for b in to_explode:
        new_tiles |= blast_tiles_for(grid, b['x'], b['y'], expl_range)

    if new_tiles:
        new_flames.append({'tiles': new_tiles, 't': expl_duration})

    return ticking, new_flames, new_tiles

