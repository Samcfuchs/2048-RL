#from multiprocessing import Pool
import torch
from torch import nn
from torch.nn import functional as F
#from torch import multiprocessing as multi
#from torch.multiprocessing import Pool
import multiprocessing as multi
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import game as core
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

gc.enable()

epsilon = 0.90 # Probability of choosing a random action
lr = 1e-6 # Gradient descent step size
#iterations = int(1e5)
batch_size = 256
gamma = 0.99
hidden_size = 64 # Size of model hidden layer
memory_size = int(1e5) # Number of moves in our training corpus each epoch
games_per_epoch = 100
training_iterations = int(1000) # Number of batches to train on
epochs = 100
device = 'cpu'

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.rand((x.shape[0], 4))


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.classes = 11

        self.flat = nn.Flatten()
        #self.layer1 = nn.Linear(16, hidden_size)
        #self.layer2 = nn.Linear(hidden_size, 4)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(self.classes, 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1, 1)
        self.pool = nn.MaxPool2d(3,1,1)

        self.linear1 = nn.Linear(64, hidden_size)
        self.out = nn.Linear(hidden_size, 4)

        self.apply(self.init_weights)
    
    def forward(self, x):

        x = self.preprocess(x)
        #print(x.shape)
        z1 = self.relu(self.conv1(x))
        z2 = self.conv2(z1)
        z3 = self.relu(self.pool(z2))
        z4 = self.relu(self.linear1(z3.reshape(-1,64)))
        z5 = self.out(z4)
        #print(z2.shape)

        return z5

    def preprocess(self, x):
        x = x.reshape(-1,4,4)

        x_log = torch.log2(x + (x==0).int())
        onehot = F.one_hot(x_log.long(), num_classes=self.classes)

        # Transpose into (Batch, Channel, Row, Column) order
        output = np.transpose(onehot, axes=[0,3,1,2])

        return output.float()
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


def ez(x):
    return torch.tensor([[0,0,1,0]])

all_actions = []
def play_turn(model, game, eps=epsilon):

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
        reward = game.score - 1e6

    action_oh = torch.zeros((1,4))
    action_oh[0][action] = 1

    new_state = torch.from_numpy(game.grid).unsqueeze(0).float().clone()

    # Save this transition to our replay memory
    return (state, action_oh, reward, new_state, terminal), flubs



def train(replay_memory, model, optimizer, criterion):
    #assert not (replay_memory[0][3] == replay_memory[1][3]).all()

    # Sample a random mini-batch to train on
    minibatch = random.sample(replay_memory, k=batch_size)

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
    mask = 1 - torch.tensor([d[4] for d in minibatch], dtype=float).to(device)
    residuals = gamma * torch.max(outputs, dim=1).values
    assert len(residuals) == batch_size
    y_batch = rewards + (mask * residuals)

    # Calculate the Q-value
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


def test(model, n_games):
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
        'move_distribution': moves / total_turns,
        'turns': total_turns,
        'highest_tile': np.log2(highest_tile)
    }
    
    return stats


if __name__ == "__main__":

    model = DQN()
    baseline = Baseline()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(model)

    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)


    #print("Trainable parameters 1:", count_params(model.conv1))
    #print("Convolution weights shape:", model.conv1.weight.shape)
    print("Trainable parameters all:", count_params(model))
    print()

    stats = test(model, n_games=100)
    print(f"Untrained: {stats}")
    #print(torch.bincount(torch.cat(all_actions)))

    plt.bar(['u','d','l','r'], stats['move_distribution'])
    plt.ylim(top=1)
    plt.grid(axis='y')
    plt.title("Move distribution of untrained model")
    plt.savefig("img/untrained_move_dist.png")

    stats = test(baseline, n_games=100)
    print(f"Baseline: {stats}")

    losses = []
    statses = []
    model.train()
    #multi.set_sharing_strategy('file_system')

    print("Begin training")

    for e in tqdm(range(epochs), ncols=100):

        replay_memory = []


        def play_game(_):
            memory = []

            game = core.Game()
            game.start()

            while not game.is_over():
                mem, _ = play_turn(ez, game, epsilon)

                memory.append(mem)
            
            return memory

        """
        # Attempt multithreading
        with Pool(6) as p:
            entries = p.map(play_game, list(range(200)))
            replay_memory += entries
            #for game in entries:
            #    replay_memory += game
        
        #print(len(replay_memory))

        replay_memory = [move for game in replay_memory for move in game]
        #print(len(replay_memory))

        """

        game = core.Game()
        game.start()
        for i in (range(memory_size)):

            # Start a new game if necessary
            if game.is_over():
                game.start()

            mem, _ = play_turn(model, game)
            replay_memory.append(mem)

            # HWM decrease the value of epsilon to make our algorithm more exploitative

        # Train the model a bunch of times
        for i in (range(training_iterations)):
            l = train(replay_memory, model, optimizer, criterion)
            losses.append(l.detach().item())
        
        stats = test(model, n_games=100)
        statses.append(stats)

        del replay_memory
        gc.collect()

    plt.figure()
    plt.plot(range(epochs), [s['avg_score'] for s in statses])
    plt.ylabel("Average score")
    plt.xlabel("Epoch")
    plt.title("Score")
    plt.grid()
    plt.savefig("img/scores.png")

    plt.figure()
    plt.bar(['u','d','l','r'], stats['move_distribution'])
    plt.ylim(top=1)
    plt.title("Move distribution for trained model")
    plt.grid(axis='y')
    plt.savefig("img/trained_move_dist.png")

    plt.figure()
    plt.plot(range(epochs), [s['miss_rate'] for s in statses])
    plt.title('Miss Rate')
    plt.xlabel("Epochs")
    plt.grid()
    plt.ylabel("Miss rate")
    plt.savefig("img/miss_rate.png")

    plt.figure()
    plt.plot(range(epochs), [s['avg_turns'] for s in statses])
    plt.title('Turns Played')
    plt.xlabel("Epochs")
    plt.grid()
    plt.ylabel("Turns Played")
    plt.savefig("img/turns_played.png")

    plt.figure()
    plt.plot(range(epochs), [s['highest_tile'] for s in statses])
    plt.title('Highest tile')
    plt.xlabel("Epochs")
    plt.grid()
    plt.ylabel("Turns Played")
    plt.savefig("img/turns_played.png")

    print("="*10)
    print("Trained:", stats)

    torch.save(model.state_dict(), "models/trained.pth")
    print("Saved model")
