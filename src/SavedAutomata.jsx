const saved_automata = {
    gol: 
    `
class Automaton:

    def __init__(self, size):
        # Initialize with random binary state (0 or 1)
        self.grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15])
        self.radius = 12
        self.density =  0.6
    
    def draw(self):
        # Calculate the number of live neighbors for each cell using convolution
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Count neighbors using convolution with periodic boundaries
        neighbors = sum(
            np.roll(np.roll(self.grid, i, 0), j, 1) * kernel[i+1, j+1]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            if not (i == 0 and j == 0)
        )
        
        # Apply Conway's Game of Life rules:
        # 1. Live cell with 2 or 3 neighbors survives
        # 2. Dead cell with exactly 3 neighbors becomes alive
        # 3. All other cells die or stay dead
        self.grid = ((neighbors == 3) | (self.grid & (neighbors == 2))).astype(int)
        
        return self.grid
      
    def spray(self, x, y):
        # Define the square region bounds with periodic boundary handling
        size = self.grid.shape[0]
        r = self.radius
        
        # Create a grid of coordinates relative to center
        y_coords, x_coords = np.ogrid[-r:r+1, -r:r+1]
        # Calculate distances from center for each point
        distances = np.sqrt(x_coords**2 + y_coords**2)
        # Create circular mask with random density
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.density)
        
        # Apply spray pattern with periodic boundaries
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if spray_pattern[i+r, j+r]:
                    # Use modulo for periodic boundaries
                    new_x = (x + i) % size
                    new_y = (y + j) % size
                    self.grid[new_x, new_y] = 1

auto = Automaton(300)

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)
    `,
    

    lenia: 
    `
class Automaton:

    def __init__(self, size):
        self.size = size
        # Initialize with random state between 0 and 1
        self.grid = np.random.uniform(0, 1, size=(size, size)) * np.random.choice([0, 1], size=(size, size), p=[0.9, 0.1])
        self.radius = 12
        self.density = 0.6
        
        # Lenia parameters
        self.dt = 0.1
        self.R = 13
        self.T = 0.15
        self.sigma = 0.015
        
        # Pre-compute kernel in frequency domain
        self.kernel = self._create_kernel()
        # Pad kernel to match grid size and compute FFT
        self.kernel_fft = np.fft.fft2(self._pad_kernel())
    
    def _create_kernel(self):
        # Create kernel using broadcasting
        x = np.linspace(-self.R, self.R, 2*self.R + 1)
        dists = np.sqrt(x[:, None]**2 + x[None, :]**2) / self.R
        
        # Create ring-shaped kernel
        kernel = np.exp(-((dists - 0.5) ** 2) / (2 * 0.15**2))
        kernel[dists > 1] = 0
        
        # Normalize
        return kernel / np.sum(kernel)
    
    def _pad_kernel(self):
        # Pad kernel to match grid size
        kernel_size = 2 * self.R + 1
        padded = np.zeros((self.size, self.size))
        start = (self.size - kernel_size) // 2
        padded[start:start+kernel_size, start:start+kernel_size] = self.kernel
        return np.fft.fftshift(padded)
    
    def _growth_function(self, x):
        # Vectorized growth function
        return np.exp(-((x - self.T) ** 2) / (2 * self.sigma**2)) * 2 - 1
    
    def draw(self):
        # Compute convolution using FFT
        fft_grid = np.fft.fft2(self.grid)
        potential = np.real(np.fft.ifft2(fft_grid * self.kernel_fft))
        
        # Update grid using vectorized operations
        self.grid = np.clip(self.grid + self.dt * self._growth_function(potential), 0, 1)
        return self.grid
    
    def spray(self, x, y):
        # Vectorized spray implementation
        r = self.radius
        
        # Create mask using broadcasting
        i, j = np.ogrid[-r:r+1, -r:r+1]
        distances = np.sqrt(i**2 + j**2)
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.density)
        
        # Calculate indices with periodic boundaries
        x_indices = (x + i) % self.size
        y_indices = (y + j) % self.size
        
        # Update grid in one operation
        self.grid[x_indices, y_indices] = np.where(spray_pattern, 1, self.grid[x_indices, y_indices])

auto = Automaton(300)

def main():
  return auto.draw()

def spray(x, y):
  auto.spray(x, y)

    `
}

export default saved_automata