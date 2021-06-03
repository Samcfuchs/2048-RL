from audioop import avg
from torch import double, nn
import torch
import numpy as np
import matplotlib.pyplot as plt
import game as core
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc

gc.enable()

epsilon = 0.95 # Probability of choosing a random action
lr = 1e-6 # Gradient descent step size
#iterations = int(1e5)
batch_size = 256
gamma = 0.99
hidden_size = 64 # Size of model hidden layer
memory_size = int(1e5) # Number of moves in our training corpus each epoch
training_iterations = int(1e4) # Number of batches to train on
epochs = 10

class Baseline(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.rand((x.shape[0], 4))


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(16, hidden_size)
        #self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 4)
        self.relu = nn.ReLU()

        self.apply(self.init_weights)
    
    def forward(self, x):

        #x_prep = torch.log2(x.reshape(-1, 16))
        x_prep = (x.reshape(-1, 16))
        z1 = self.relu(self.layer1(x_prep))
        #z2 = self.relu(self.layer2(z1))
        z3 = self.layer3(z1)
        #out = self.activation(z3)
        #assert torch.argmax(z3) == 0
        return z3
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

model = DQN()
baseline = Baseline()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

game = core.Game()
game.start()

def train(replay_memory):
    #assert not (replay_memory[0][3] == replay_memory[1][3]).all()

    # Sample a random mini-batch to train on
    minibatch = random.sample(replay_memory, k=batch_size)

    # Unpack mini-batch into state, action, reward, and new-state vectors
    prestates = torch.cat([d[0] for d in minibatch])
    actions = torch.cat([d[1] for d in minibatch])
    rewards = torch.cat([torch.tensor([d[2]]) for d in minibatch])
    poststates = torch.cat([d[3] for d in minibatch])

    # HWM move those vectors onto the GPU to train

    # Get a batch of new actions based on the new-states
    outputs = model(poststates.float())
    #new_actions = torch.argmax(outputs, dim=1)
    #print("New actions shape:", new_actions.shape)


    y_batch = torch.zeros(len(minibatch), dtype=float)
    for i in range(len(minibatch)):
        if minibatch[i][4]:
            # This is a game-end state
            y_batch[i] = rewards[i]
        else:
            #assert gamma * torch.max(outputs[i]) > 0
            y_batch[i] = rewards[i] + gamma * torch.max(outputs[i])

    # y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
    #                           else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
    #                           for i in range(len(minibatch))))


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

def play_turn(model, game, eps=epsilon):

    # Get the game state from the game and choose a move
    state = torch.from_numpy(game.grid).unsqueeze(0).float().clone()
    output = model(state)

    action = torch.argmax(output)

    # Sometimes just choose a random move instead
    do_random = np.random.random() < eps
    if do_random:
        action = np.random.randint(0,4)

    # Get the next state and calculate the reward
    new_state, illegal, terminal = game.turn4(action)

    # OPTIMIZE
    flubs = 0
    while illegal:

        #print("oopsie!")
        # This board has NOT been populated yet.
        action = np.random.randint(0,4)
        new_state, illegal, terminal = game.turn4(action)
        flubs += 1
    #print("FIXED IT!")

    reward = game.score

    if terminal:
        reward = game.score - 1e6

    action_oh = torch.zeros((1,4))
    action_oh[0][action] = 1

    new_state = torch.from_numpy(game.grid).unsqueeze(0).float().clone()
    #print(f"new_state: {new_state}")



    # Save this transition to our replay memory
    return (state, action_oh, reward, new_state, terminal), flubs

def test(model, n_games):
    game = core.Game()
    total_score = 0
    total_illegals = 0
    total_turns = 0
    moves = torch.zeros(4)
    for g in range(n_games):

        game.start()

        while not game.is_over():
            (_,action,_,_,_), illegals = play_turn(model, game, eps=0)

            moves[torch.argmax(action)] += 1
            total_illegals += illegals
            total_turns += 1
        
        total_score += game.score
    
    stats = {
        'avg_score': total_score / n_games,
        'avg_turns': total_turns / n_games,
        'miss_rate': total_illegals / total_turns,
        'misses_per_game': total_illegals / n_games,
        'move_distribution': moves / total_turns,
        'turns': total_turns
        #'ups': moves[0]
    }
    
    return stats

stats = test(model, n_games=100)
print(f"Untrained: {stats}")

stats = test(baseline, n_games=100)
print(f"Baseline: {stats}")

losses = []
statses = []
for e in tqdm(range(epochs)):

    replay_memory = []

    for i in (range(memory_size)):

        # Start a new game if necessary
        if game.is_over():
            game.start()

        mem, _ = play_turn(model, game)
        replay_memory.append(mem)

        # HWM decrease the value of epsilon to make our algorithm more exploitative

    # Train the model a bunch of times
    for i in (range(training_iterations)):
        l = train(replay_memory)
        losses.append(l.detach().item())
    
    stats = test(model, n_games=100)
    statses.append(stats)

    del replay_memory
    gc.collect()

plt.plot(range(epochs), [s['avg_score'] for s in statses])
plt.ylabel("Average score")
plt.xlabel("Epoch")
plt.grid()
plt.savefig("img/scores.png")

plt.figure()
plt.bar(['u','d','l','r'], stats['move_distribution'])
plt.ylim(top=1)
plt.title("Testing move distribution for trained model")
plt.grid('x')
plt.savefig("img/trained_move_dist.png")

plt.figure()
plt.plot(range(epochs), [s['miss_rate'] for s in statses])
plt.title('Miss Rate')
plt.xlabel("Epochs")
plt.grid()
plt.ylabel("Miss rate")
plt.savefig("img/miss_rate.png")

print("="*10)
print(stats)
