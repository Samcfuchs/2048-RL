# Learning 2048

I design a neural network that uses reinforcement learning to select ideal moves
in the popular game 2048.

## Baseline testing

We create a baseline model which chooses a direction randomly. Scores tend to
fluctuate a lot with these models, presumably because it's possible for the game
to end very quickly, or just as likely very slowly. 

## Evaluation Techniques

It's useful to keep an eye on the distribution of moves--we want to notice when
the model is converging to prefer the same move every time, and when it doesn't.
I compare the move distribution of the untrained model to its move distribution
after training to see whether it has changed its behavior.

In these preliminary stages, I often set an arbitrary reward for the model
choosing one of the directions--this is a much simpler behavior for the model to
learn than the actual game, and we can tune the model's efficiency and examine
its behavior in response to this simple reward system before moving to a much
more complex one.

We also deviate from the standard 2048 ruleset by implementing a penalty for
attempting an illegal move. This way, we intend that the learner will gradually
learn which moves are and which moves are not legal--this is also a lower-order
behavior for the model to learn, and so the rate at which the model selects
these illegal moves is another way to determine how the network is developing.

## Model Architecture

My initial design is a simple feedforward network with a single hidden layer. I
take in the raw numerical values of the board and flatten that matrix to form an
input vector of size 16. The output of the network is a vector of four values
which correspond to the four possible moves: up, down, left, and right. I
select the move that the model has scored most highly and enact it in the game.

In some cases, a move is impossible--this occurs if there are no tiles that can
move in that direction. In this case, we select the next best-rated move. If
there are no legal moves, then the game has ended.

### Improvements

This initial architecture is probably not very good. To improve upon it, some
key features should be implemented.

We should prefer convolutional layers to linear ones, as they'll be better
equipped to handle the dynamic nature of positioning. However, I can't eschew
linear layers entirely, since the placement of tile groups on the board is very
important to higher-level strategy.

The input encoding is far too simple. We should at minimum prefer a labeling
system (i.e. $log_2x$) that will eliminate the order-of-magnitude differences
between different tiles. One-hot encoding of the inputs is another option which
we can consider, perhaps combining it with some really weird convolutional layer
shapes.

It's hard to say whether these flaws will limit our model's performance on
baseline tasks, or whether they should be reserved for later development.

## Some Results

With these parameters, I see a steady increase in average game scores for my
model:

- epsilon = 0.95 # Probability of choosing a random action
- lr = 1e-6 # Gradient descent step size
- batch_size = 256
- gamma = 0.99
- hidden_size = 256 # Size of model hidden layer
- memory_size = int(1e5) # Number of moves in our training corpus each epoch
- training_iterations = int(2e3) # Number of batches to train on
- epochs = 60

This is with just one hidden layer.
