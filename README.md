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
system (i.e. $$log_2x$$) that will eliminate the order-of-magnitude differences
between different tiles. One-hot encoding of the inputs is another option which
we can consider, perhaps combining it with some really weird convolutional layer
shapes.

It's hard to say whether these flaws will limit our model's performance on
baseline tasks, or whether they should be reserved for later development.

This feedforward network has 5380 parameters:
- layer 1: 4352
- layer 2: 1028

## Convolutional Architecture

I compose a new convolutional architecture to approach the problem. Instead of
representing with labels, I create a one-hot vector categorizing each tile as
one of the 11 possible powers of two. This gives us an input vector with the
shape 11x4x4, on which we perform 2-dimensional convolutions with a 3x3 kernel.
We leave the image size unchanged (with a stride of 1) for both layers of
convolution and pooling.

After 100 epochs of training, we see a significant increase in performance on
the one-directional task. The model learns to favor two directions, the scoring
one and another. This shows that not only is the model learning to select the
highest-scoring direction, it also learns to extend the game so that it can
enter more moves in that direction. This behavior, of extending the game, is a
huge step forward for the model. The "real" scoring behavior of 2048 is quite
sparse, but these results give me reason to believe that this model can improve.

## SmartCNN

Because diagonal relationships aren't very important in 2048, we might see
better performance by focusing exclusively on the salient vertical and
horizontal relationships. This model develops its understanding of the board
state more as a network of connected nodes rather than a coherent grid.

My feature set is similar to that of the previous architecture, but with one
added feature to capture the boolean "emptiness" of the board: a mask
representing which squares are occupied and which are empty. This makes it
easier for the model to understand where the next tile may appear. This model
also performs a normalization step which places the highest-value corner tile in
the upper left-hand corner of the board. This allows the model some invariance
with respect to *which* corner it accumulates tiles in. Because the game is
rotationally symmetric, we can expect the model to learn patterns more quickly
under normalized conditions.

This CNN uses two separate convolutional layers: one with a (1x2) kernel to
capture horizontal relationships, and one with a (2x1) kernel to capture
relationships between vertical tiles, each with four output features. With these
smaller kernels, I no longer use any padding, which eliminates another weakness
of the previous model. I concatenate these two (12x4) matrices to create the
feature vector which is processed by two linear layers. One of the key
advantages of CNN architectures is invariance to translation, but in this case,
we want to preserve locational relationships in order for the model to make more
circumspect decisions, so I omit pooling steps and other aggregations

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

## Resources

- https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7
- https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

- https://www.mit.edu/~adedieu/pdf/2048.pdf

- https://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/
- https://cs.uwaterloo.ca/~mli/zalevine-dqn-2048.pdf
