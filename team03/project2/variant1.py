# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../team03')
from team3qchar import QChar 

qchar = QChar(
    "me", # name
    "C",  # avatar
    0, 0  # position
)

# Create the game
while True:
    g = Game.fromfile('map.txt')

    g.add_character(qchar)

    g.go(1)
    pass
