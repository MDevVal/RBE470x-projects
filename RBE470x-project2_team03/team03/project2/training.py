# This is necessary to find the main code
import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearningcharacter import QLearningCharacter
from monsters.stupid_monster import StupidMonster

for _ in range(1000):
    # Create the game
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                                "S",      # avatar
                                3, 9      # position
    ))
    # TODO Add your character
    g.add_character(QLearningCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))

    # Run!
    g.go(1) # skip pause


    print("FINAL SCORE \n --------------------------------------------\n",g.world.scores["me"])
    # if g.wrld.scores["me"] > 0:
    #     break
    # print("Weights: ", )

