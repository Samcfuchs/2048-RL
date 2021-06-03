import numpy as np
# Convention: axis 0 is rows, axis 1 is columns

class Game(object):

    penalty = 32

    def __init__(self, size=4, four_rate=0.10):
        self.size = size
        self.four_rate = four_rate
        self.grid = np.zeros((size, size), dtype=int)

    """ Determine the random value of a new tile """
    def new(self):
        r = np.random.random()
        if r < self.four_rate:
            return 4
        else:
            return 2

    """ Set up the game and populate the board """
    def start(self):
        # Reset the board
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.game_over = False
        self.illegal = False # indicates whether the most recent move was illegal
        self.score = 0

        # Generate some coordinates
        # TODO What if the coordinates are the same?
        # I think that's handled now
        squares = np.random.randint(0, self.size, size=(2,2))
        while (squares[0] == squares[1]).all():
            squares = np.random.randint(0, self.size, size=(2,2))

        self.grid[tuple(squares[0])] = self.new()
        self.grid[tuple(squares[1])] = self.new()


    def is_over(self):
        """ Determine whether there are empty spaces remaining on the board """
        return not (self.grid == 0).any()


    # Axis: what axis to move on    
    # L/R : Rows : axis 0
    # U/D : Cols : axis 1
    # 
    # Direction: which way to go
    # 1: D/R
    # -1: U/L
    """ Execute the specified move """
    arrow = { "l": (0, -1), "r": (0, 1), "u": (1, -1), "d": (1, 1) }
    def move(self, axis, direction):
        #print("Update Matrix")
        old = self.grid.copy()
        g = self.grid.T if axis else self.grid
        for i,row in enumerate(g):
            row = row[::direction]
            #print(row)

            # Merge everything right
            merged = self.merge_left(row[::-1])[::-1]

            # Now shift everything right
            r = [0] * (self.size - len(merged)) + merged
            
            g[i] = r[::direction]

        self.grid = g.T if axis else g

        self.illegal = (self.grid == old).all()
        
        return self.grid, self.illegal
    

    moves = [arrow['u'], arrow['d'], arrow['l'], arrow['r']]
    def move4(self, n):
        """ 
        A wrapper for move that takes an integer in range(0,4) representing one
        of the moves [up, down, left, right]
        """
        return self.move(*Game.moves[n])
        
    
    def populate(self):
        """ 
        Add a tile in an empty space. If no such space is available, return the
        current grid and the value True to indicate that the game is over.
        """
        self.game_over = self.is_over()
        if self.game_over:
            return self.grid, self.game_over

        empty_indices = np.transpose((self.grid==0).nonzero())

        assert len(empty_indices) != 0
        
        index = np.random.randint(0, empty_indices.shape[0])
        selection = empty_indices[index]

        self.grid[tuple(selection)] = self.new()

        return self.grid, self.game_over


    #@staticmethod
    """ Merge the elements of a list toward the left. Removes zeros """
    def merge_left(self, row):
        skip = False
        merged = []
        row = [v for v in row if v]
        for i in range(len(row)):
            if skip or row[i] == 0:
                #print("Skipped", i)
                skip = False
                continue

            if i == len(row)-1:
                #print("Left", i)
                merged.append(row[i])
                continue

            if row[i] == row[i+1]:
                merged.append(row[i]*2)
                skip = True
                self.score += row[i] * 2
                #print(f"Merged {i} and {i+1}")
                continue
            
            #print("Left", i)
            merged.append(row[i])

        return merged
    
    """ Display the board in the terminal """
    def display(self):
        print("="*10)
        print(self.grid)
        print("="*10)
    
    """ Execute a full turn """
    def turn(self, move):
        self.move(*move)
        self.populate()
        self.display()
    
    def turn4(self, n):
        """ 
        Execute a move as in move4. If it is a valid move, then populate the
        board. Return the board (whether or not it is changed), whether the move
        was valid, and whether this is a terminal game state.
        """
        _, illegal = self.move4(n)

        if not illegal:
            _, is_over = self.populate()
        
        if n==2:
            self.score += self.penalty
        
        return self.grid, illegal, self.is_over()


if __name__ == "__main__":
    arrow = { "l": (0, -1), "r": (0, 1), "u": (1, -1), "d": (1, 1) }

    game = Game()
    game.start()
    game.display()
    game.turn(arrow['l'])
