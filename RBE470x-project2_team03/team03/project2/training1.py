# This is necessary to find the main code
import sys
import time
import numpy as np
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

# Import necessary stuff
from game import Game

# TODO This is your code!
sys.path.insert(1, '../teamNN')
from testcharacter import TestCharacter
from qlearningcharacter import QLearningCharacter
from monsters.stupid_monster import StupidMonster
record = []
scores=[]
for _ in range(1000):
    # Create the game
    g = Game.fromfile('map.txt')
    # g.add_monster(StupidMonster("stupid", # name
    #                             "S",      # avatar
    #                             3, 9      # position
    # ))
    # TODO Add your character
    g.add_character(QLearningCharacter("me", # name
                                "C",  # avatar
                                0, 0  # position
    ))

    # Run!
    g.go(1) # skip pause


    print("FINAL SCORE \n --------------------------------------------\n",g.world.scores["me"])

    scores.append(g.world.scores["me"])
    if g.world.scores["me"] < 0:
        record.append(0)
    else:
        record.append(1)
    percentage=[np.mean(record)*100,np.mean(scores)]
    print("Percentage of wins: ", percentage)
    sav=np.concatenate((percentage,record))
    print(sav)
    np.savetxt("record1.txt", sav, fmt='%d')

    # time.sleep(1)

