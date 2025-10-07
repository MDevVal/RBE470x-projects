# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
import random
from game import Game
from monsters.selfpreserving_monster import SelfPreservingMonster

# TODO This is your code!
sys.path.insert(1, '../team03')
from team3qchar import QChar 

# Create the game
random.seed() # TODO Change this if you want different random choices
g = Game.fromfile('map.txt')
g.add_monster(SelfPreservingMonster("selfpreserving", # name
                                    "S",              # avatar
                                    3, 9,             # position
                                    1                 # detection range
))

qchar = QChar("me", # name
              "C",  # avatar
              0, 0  # position
              )

qchar.load_model("variant_3.pt")

g.add_character(qchar)

# Run!
g.go(1)

qchar.save_model("variant_3.pt")

