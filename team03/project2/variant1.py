import sys
sys.path.insert(0, '../../bomberman')
sys.path.insert(1, '..')

from game import Game

sys.path.insert(1, '../team03')
from team3qchar import QChar 

qchar = QChar("me", # name
                                  "C",  # avatar
                                  0, 0  # position
) 

g = Game.fromfile('map.txt')

g.add_character(qchar)

g.go(1)
