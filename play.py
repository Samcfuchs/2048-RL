import game as core
import numpy as np

arrow = { "l": (0, -1), "r": (0, 1), "u": (1, -1), "d": (1, 1) }

board = np.array([[0,0,0,2],
                  [0,2,0,0],
                  [0,0,0,0],
                  [0,0,0,0]])

game = core.Game()

game.start()

# HACK
game.grid = board

game.display()

game.move4(1)

game.display()

game.populate()
game.display()
