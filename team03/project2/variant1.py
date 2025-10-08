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

i = 0
while True:
    g = Game.fromfile('map.txt')
    

    g.add_character(qchar)
    qchar.x = 0
    qchar.y = 0

    g.go(1)
    
    i += 1
    if i % 100 == 0:
        qchar.save_model(f"variant_{i}.pt")
        qchar.reset_episode()

    print(f"Iteration {i}")
