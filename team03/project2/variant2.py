# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

import random
from game import Game
from monsters.stupid_monster import StupidMonster

sys.path.insert(1, '../team03')
from team3qchar import QChar 

random.seed()

g = Game.fromfile('map.txt')
g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 9      # position
                            ))

qchar = QChar("me", # name
              "C",  # avatar
              0, 0  # position
              )

g.add_character(qchar)

g.go(1)
