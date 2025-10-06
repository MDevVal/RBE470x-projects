# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../')

# Uncomment this if you want the empty test character
#from team3char import Team3Char

# Uncomment this if you want the interactive character
efrom interactivecharacter import InteractiveCharacter

# Uncomment if you trust Benni
from team3char import Team3Char

# Create the game
g = Game.fromfile('map.txt')

# TODO Add your character

#Uncomment this if you want the test character
g.add_character(Team3Char("me", # name
                              "C",  # avatar
                              0, 0  # position
))

# Uncomment this if you want the interactive character
#g.add_character(InteractiveCharacter("me", # name
#                                     "C",  # avatar
#                                     0, 0  # position
#))

# Run!

# Use this if you want to press ENTER to continue at each step
# g.go(0)

# Use this if you want to proceed automatically
g.go(1)
