import game as core
import numpy as np
import models
import torch
from train import hidden_size
import sys

# Run a model through a game and see how it plays.

folder = sys.argv[1]
filename = f"results/{folder}/trained.pth"

model = models.DQN(hidden_size)
model.load_state_dict(torch.load(filename))

game = core.Game()

game.start()

dirs = ['u','d','l','r']

while not game.is_over():
    game.display()
    print("Score:", game.score)

    x = torch.tensor(game.grid).float()
    output = model(x)
    move = torch.argmax(output)

    _, illegal, _ = game.turn4(move)

    flubs = 0
    while illegal:
        output[0, move] = torch.min(output) - 1
        move = torch.argmax(output)
        #action = np.random.randint(0,4)
        _, illegal, _ = game.turn4(move)
        flubs += 1

    print("Move:", dirs[move])
    if flubs: print(flubs, "illegal moves tried")

    #input()

