# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

import random
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster

sys.path.insert(1, '../team03')
from team3qchar import QChar

random.seed()
g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
))

qchar = QChar("me", # name
              "C",  # avatar
              0, 0  # position
              )

g.add_character(qchar)

g.go(1)
