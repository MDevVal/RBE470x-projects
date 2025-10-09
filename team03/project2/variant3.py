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
g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
))

qchar = QChar("me",
              "C",
              0, 0
              )

g.add_character(qchar)

g.go(1)
