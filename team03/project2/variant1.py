# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../team03')
from team3qchar import QChar 
qchar = QChar("me", # name
                              "C",  # avatar
                              0, 0  # position
)

qchar.load_model("variant_1.pt")

# Create the game
g = Game.fromfile('map.txt')

# TODO Add your character
g.add_character(QChar("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Run!
g.go(1)

qchar.save_model("variant_1.pt")
