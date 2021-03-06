#!/home/sam/miniconda3/envs/datasci/bin/python

import json
import os
import torch
from torch import nn
import numpy as np
import game as core
import random
from tqdm import tqdm
import gc
import models

gc.enable()

lr = 1e-6 # Gradient descent step size
#iterations = int(1e5)
batch_size = 256
losing_cost = 0 # cost incurred on the end game state
gamma = 0.90
memory_size = int(1e4) # Number of moves in our training corpus each epoch
training_iterations = int(1e3) # Number of batches to train on
# Epsilon is the probability of choosing a random action
epsilon_decay = 0.005 # Rate of epsilon's exponential decay
epsilon_fn = lambda epoch: (1-epsilon_decay)**epoch
epochs = 100
test_interval = 5 # How often to run a test and output results

device = 'cpu'

in_folder = "results/jul5/"
#in_folder = ""
if in_folder:
    model_path = in_folder + "trained.pth"
    stats_file = in_folder + "stats.json"
else:
    model_path = ""
    stats_file = ""

out_folder = 'results/jul5/'
stats_file_out = out_folder + "stats.json"


try:
    os.mkdir(out_folder)
    os.mkdir(out_folder + "models/")
except FileExistsError:
    pass


def ez(x):
    return torch.tensor([[0,0,1,0]])

all_actions = []
def play_turn(model: nn.Module, game: core.Game, eps=epsilon_fn(0)):

    # Get the game state from the game and choose a move
    state = torch.from_numpy(game.grid).unsqueeze(0).float().clone().to(device)
    output = model(state)

    action = torch.argmax(output)
    #all_actions.append(action.unsqueeze(0))

    # Sometimes just choose a random move instead
    do_random = np.random.random() < eps
    if do_random:
        action = np.random.randint(0,4)

    # Get the next state and calculate the reward
    new_state, illegal, terminal = game.turn4(action)

    # OPTIMIZE
    flubs = 0
    while illegal:
        #print("Looping")
        # This board has NOT been populated yet.
        # Choose the next best action
        output[0, action] = torch.min(output) - 1
        action = torch.argmax(output)
        #action = np.random.randint(0,4)
        new_state, illegal, terminal = game.turn4(action)
        flubs += 1
    #print("Escaped")

    reward = game.score

    if terminal:
        reward = game.score - losing_cost

    action_oh = torch.zeros((1,4))
    action_oh[0][action] = 1

    new_state = torch.from_numpy(game.grid).unsqueeze(0).float().clone()

    # Save this transition to our replay memory
    return (state, action_oh, reward, new_state, terminal), flubs


def train(replay_memory, model, optimizer, criterion):
    """ Select a single batch from the replay memory and train on that batch """
    #assert not (replay_memory[0][3] == replay_memory[1][3]).all()

    # Sample a random mini-batch to train on
    minibatch = random.sample(replay_memory, k=int(batch_size))

    # Unpack mini-batch into state, action, reward, and new-state vectors
    prestates = torch.cat([d[0] for d in minibatch]).to(device)
    actions = torch.cat([d[1] for d in minibatch]).to(device)
    rewards = torch.cat([torch.tensor([d[2]]) for d in minibatch]).to(device)
    poststates = torch.cat([d[3] for d in minibatch]).to(device)

    # HWM move those vectors onto the GPU to train

    # Get a batch of new actions based on the new-states
    outputs = model(poststates.float())
    #new_actions = torch.argmax(outputs, dim=1)
    #print("New actions shape:", new_actions.shape)

    # TODO optimize this somehow
    # Calculate scores for each move
    mask = 1 - torch.tensor([d[4] for d in minibatch], dtype=float).to(device)
    q_next = gamma * torch.max(outputs, dim=1).values
    assert len(q_next) == batch_size
    y_batch = rewards + (mask * q_next)

    # Calculate the Q-value
    # Q describes how confident the prediction is
    outputs = model(prestates)
    q = torch.sum(outputs * actions, dim=1, dtype=float)

    # Zero the gradients
    optimizer.zero_grad()

    y_batch = y_batch.detach()

    loss = criterion(q, y_batch)

    loss.backward()
    optimizer.step()

    del prestates
    del poststates
    del actions
    del rewards
    del minibatch
    del outputs

    return loss


def test(model: nn.Module, n_games):
    model.eval()
    game = core.Game()
    total_score = 0
    total_illegals = 0
    total_turns = 0
    moves = torch.zeros(4)
    highest_tile = 2
    for g in range(n_games):

        game.start()

        while not game.is_over():
            (_,action,_,_,_), illegals = play_turn(model, game, eps=0)

            moves[torch.argmax(action)] += 1
            total_illegals += illegals
            total_turns += 1

        total_score += game.score
        highest_tile = max(np.max(game.grid), highest_tile)


    stats = {
        'avg_score': total_score / n_games,
        'avg_turns': total_turns / n_games,
        'miss_rate': total_illegals / total_turns,
        'misses_per_game': total_illegals / n_games,
        'move_distribution': (moves / total_turns).tolist(),
        'turns': total_turns,
        'highest_tile': np.log2(highest_tile)
    }

    return stats


if __name__ == "__main__":

    model = models.SmartCNN()
    # Import saved weights if appropriate
    previous_epochs = 0

    if model_path:
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model weights")
        with open(stats_file, 'r') as f:
            stats = json.load(f)
            assert type(stats) is list
        previous_epochs = stats[-1]['epoch']
        print(f"Loaded {previous_epochs} previous stats")
    else:
        stats_untrained = test(model, n_games=100)
        stats_untrained['epoch'] = -1
        print(f"Untrained: {stats_untrained}")
        stats = [stats_untrained]

    model.to(device)

    """
    g = core.Game()
    g.start()

    x = torch.from_numpy(g.grid).float()
    x = torch.stack((x,x), dim=0)
    output = model(x)
    print(output)
    raise KeyboardInterrupt
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(model)

    print("# of trainable parameters:", models.count_params(model))
    print()

    # baseline = models.Baseline()
    # stats = test(baseline, n_games=100)
    # print(f"Baseline: {stats}")

    #losses = []
    model.train()

    print("Begin training")

    def play_game(_):
        memory = []

        game = core.Game()
        game.start()

        while not game.is_over():
            mem, _ = play_turn(ez, game)

            memory.append(mem)

        return memory

    for e in tqdm(range(previous_epochs+1, previous_epochs+1 + epochs), ncols=70):
        try:

            replay_memory = []

            game = core.Game()
            game.start()
            for i in (range(memory_size)):

                # Start a new game if necessary
                if game.is_over():
                    game.start()

                mem, _ = play_turn(model, game, eps=epsilon_fn(e))
                replay_memory.append(mem)

                # HWM decrease the value of epsilon to make our algorithm more exploitative

            loss = 0
            # Train the model a bunch of times
            for i in (range(training_iterations)):
                l = train(replay_memory, model, optimizer, criterion)
                loss += l

            if e % test_interval == 0:
                stat = test(model, n_games=100)
                stat['epoch'] = e
                stat['train_loss'] = (loss / training_iterations).item()
                stats.append(stat)

                if stats_file_out:
                    with open(stats_file_out, 'w') as f: json.dump(stats, f)
                
                torch.save(model.state_dict(), out_folder + f"models/trained_{e}.pth")

            del replay_memory
            gc.collect()

        except KeyboardInterrupt:
            print("Interrupt received")
            break

    # Save the stats
    if stats_file_out:
        with open(stats_file_out, 'w') as f: json.dump(stats, f)
    print("Saved stats")

    # Save the model
    torch.save(model.state_dict(), out_folder + "trained.pth")
    print("Saved model")
