import numpy as np 

class GoL:

    def __init__(self, size):
        self.grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15])
        self.radius = 12
        self.density =  0.6

        self.kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    
    def draw(self):
        
        neighbors = sum(
            np.roll(np.roll(self.grid, i, 0), j, 1) * self.kernel[i+1, j+1]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            if not (i == 0 and j == 0)
        )
        
        # live cell with 2 or 3 neighbors survives
        # dead cell with 3 neighbors becomes alive
        # everyone else dies
        self.grid = ((neighbors == 3) | (self.grid & (neighbors == 2))).astype(int)
        
        return self.grid
      
    def spray(self, x, y):
        size = self.grid.shape[0]
        r = self.radius
        y_coords, x_coords = np.ogrid[-r:r+1, -r:r+1]
        distances = np.sqrt(x_coords**2 + y_coords**2)
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.density)
        
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if spray_pattern[i+r, j+r]:
                    new_x = (x + i) % size
                    new_y = (y + j) % size
                    self.grid[new_x, new_y] = 1
