import game as core
from train import play_turn
import torch
#import multiprocessing
#from multiprocessing import Pool
from torch import multiprocessing as multi
from torch.multiprocessing import Pool

ez = lambda x: torch.tensor([0,0,1,0]).unsqueeze(0)


epsilon = 0.2

def play_game(_):
    memory = []

    game = core.Game()
    game.start()

    while not game.is_over():
        mem, _ = play_turn(ez, game, epsilon)

        memory.append(mem)

    return memory

def play_turns(n):
    memory = []
    game = core.Game()
    game.start()

    for i in range(n):
        if game.is_over():
            game = core.Game()
            game.start()
        mem, _ = play_turn(ez, game, epsilon)
        memory.append(mem)
    
    return memory


def play_multi(n):
    replay_memory = []
    # Attempt multithreading
    n_threads = 10
    with Pool(n_threads) as p:
        entries = p.map(play_turns, [n // n_threads] * n_threads)
        replay_memory += entries
        for game in entries:
            replay_memory += game

    return replay_memory

def play(n):
    replay_memory = play_turns(n)
    return replay_memory

import time

n = 10000
multi.set_sharing_strategy('file_system')

t = time.time()
m = play(n)
dur = time.time() - t
print(f"Unthreaded time: {dur}")
print(f"Time per turn: {dur / len(m)}")

t = time.time()
m = play_multi(n)
dur = time.time() - t
print(f"Unthreaded time: {dur}")
print(f"Time per turn: {dur / len(m)}")
