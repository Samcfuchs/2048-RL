#!/home/sam/miniconda3/envs/datasci/bin/python

import os
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

filename = sys.argv[1]

with open('results/' + filename + '/stats.json', 'r') as f:
    data = json.load(f)

epochs = len(data)
stats = data
img_location = f"results/{filename}/img/"

try:
    os.mkdir(img_location)
except FileExistsError:
    pass

plt.figure()
plt.bar(['u','d','l','r'], stats[0]['move_distribution'])
plt.ylim(top=1)
plt.grid(axis='y')
plt.title("Move distribution of untrained model")
plt.savefig(img_location + "untrained_move_dist.png")

plt.figure()
plt.plot(range(epochs), [s['avg_score'] for s in stats])
plt.ylabel("Average score")
plt.xlabel("Epoch")
plt.title("Score")
plt.grid()
plt.savefig(img_location + "avg_score.png")

plt.figure()
plt.bar(['u','d','l','r'], stats[-1]['move_distribution'])
plt.ylim(top=1)
plt.title("Move distribution for trained model")
plt.grid(axis='y')
plt.savefig(img_location + "move_distribution.png")

plt.figure()
plt.plot(range(epochs), [s['miss_rate'] for s in stats])
plt.title('Miss Rate')
plt.xlabel("Epochs")
plt.grid()
plt.ylabel("Miss rate")
plt.savefig(img_location + "miss_rate.png")

plt.figure()
plt.plot(range(epochs), [s['avg_turns'] for s in stats])
plt.title('Turns Played')
plt.xlabel("Epochs")
plt.grid()
plt.ylabel("Turns Played")
plt.savefig(img_location + "avg_turns.png")

plt.figure()
plt.plot(range(epochs), [s['highest_tile'] for s in stats])
plt.title('Highest tile')
plt.xlabel("Epochs")
plt.grid()
plt.ylabel("Highest Tile")
plt.savefig(img_location + "highest_tile.png")

plt.figure()
plt.plot(range(1,epochs), [s['train_loss'] for s in stats[1:]])
plt.title('Loss')
plt.xlabel("Epochs")
plt.grid()
plt.ylabel("Epoch Training Loss")
plt.savefig(img_location + "loss.png")
