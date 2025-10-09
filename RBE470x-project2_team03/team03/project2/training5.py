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
from monsters.selfpreserving_monster import SelfPreservingMonster
from monsters.stupid_monster import StupidMonster

record = []

scores=[]


for _ in range(1000):
    # Create the game
    g = Game.fromfile('map.txt')
    g.add_monster(StupidMonster("stupid", # name
                            "S",      # avatar
                            3, 5,     # position
    ))
    g.add_monster(SelfPreservingMonster("aggressive", # name
                                    "A",          # avatar
                                    3, 13,        # position
                                    2             # detection range
    ))
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
    np.savetxt("record.txt", sav, fmt='%d')
    # if g.world.scores["me"] < 0:
    #     exit()
    # if g.wrld.scores["me"] > 0:
    #     break
    # print("Weights: ", )



