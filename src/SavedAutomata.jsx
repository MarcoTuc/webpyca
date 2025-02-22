const saved_automata = {
    gol: 
    `
#### CONWAY'S GAME OF LIFE  
#### marco.tuccio95@gmail.com

#### click 'Run' to initiate the simulation
#### you can control target FPS below 'Run'

#### you can choose other pre-written automatons
#### at the right-bottom corner of the left side

class Automaton:

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


auto = Automaton(300)

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)
    
    `,
    

    lenia: 
    `

#### ONE CHANNEL LENIA
#### marco.tuccio95@gmail.com

#### click 'Run' to initiate the simulation
#### you can control target FPS below 'Run'

#### you can choose other pre-written automatons
#### at the right-bottom corner of the left side


class Automaton:

    def __init__(self, size):
        self.size = size
        self.grid = np.random.uniform(
            0, 1, 
            size=(size, size)
        ) * np.random.choice(
            [0, 1], 
            size=(size, size), 
            p=[0.9, 0.1]
        )
        self.radius = 24
        self.density = 0.6
        
        self.dt = 0.02
        self.R = 14
        self.T = 0.18
        self.sigma = 0.015
        
        self.kernel = self._create_kernel()
        self.kernel_fft = np.fft.fft2(self._pad_kernel())
    
    def _create_kernel(self):
        x = np.linspace(-self.R, self.R, 2*self.R + 1)
        dists = np.sqrt(x[:, None]**2 + x[None, :]**2) / self.R
        kernel = np.exp(-((dists - 0.5) ** 2) / (2 * 0.15**2))
        kernel[dists > 1] = 0
        return kernel / np.sum(kernel)
    
    def _pad_kernel(self):
        kernel_size = 2 * self.R + 1
        padded = np.zeros((self.size, self.size))
        start = (self.size - kernel_size) // 2
        padded[start:start+kernel_size, start:start+kernel_size] = self.kernel
        return np.fft.fftshift(padded)
    
    def _growth_function(self, x):
        return np.exp(-((x - self.T) ** 2) / (2 * self.sigma**2)) * 2 - 1
    
    def draw(self):
        fft_grid = np.fft.fft2(self.grid)
        potential = np.real(np.fft.ifft2(fft_grid * self.kernel_fft))
        self.grid = np.clip(
            self.grid + self.dt * self._growth_function(potential), 0, 1
        )
        return self.grid
    
    def spray(self, x, y):
        #### Receives an (x, y) position from the mouse
        #### and draws a ball around it when pressed  
        r = self.radius
        i, j = np.ogrid[-r:r+1, -r:r+1]
        distances = np.sqrt(i**2 + j**2)
        spray_pattern = (distances <= r) & (np.random.random(
                                            (2*r + 1, 2*r + 1)) < self.density)
        x_indices = (x + i) % self.size
        y_indices = (y + j) % self.size
        self.grid[x_indices, y_indices] = np.where(
            spray_pattern, 1, self.grid[x_indices, y_indices]
        )

auto = Automaton(500)

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)
    
    
    `,
    multilenia:
    `
#### MULTI CHANNEL LENIA
#### marco.tuccio95@gmail.com

#### click 'Run' to initiate the simulation
#### you can control target FPS below 'Run'

#### you can choose other pre-written automatons
#### at the right-bottom corner of the left side

#### If you have any idea of how to make it faster
#### by just using numpy, contact me at my email


########################### TWEAK THESE PARAMETERS

world_size = 150
number_hidden_channels = 60

brush_radius = 50
brush_density = 0.2

R = 15
dt = 0.2
power_T = 0.1
power_sigma = 0.3



class Automaton:
    
    def __init__(self):
        self.size = world_size
        self.brush_radius = brush_radius
        self.brush_density = brush_density
        self.channels = 3 + number_hidden_channels
        self.grid = np.zeros((self.size, self.size, self.channels), dtype=np.float32)
        center = self.size // 2
        radius = self.size // 4    
        self.dt = dt
        self.R = R 
        self._initialize_kernel()
        self.T = np.random.power(power_T, size=self.channels)
        self.sigma = np.random.power(power_sigma, size=self.channels)


    def _initialize_kernel(self):
        x = np.linspace(-self.R, self.R, 2*self.R + 1)
        xx, yy = np.meshgrid(x, x)
        dists = np.sqrt(xx**2 + yy**2) / self.R
        kernel = np.exp(-((dists - 0.5) ** 2) / 0.045)
        kernel[dists > 1] = 0
        kernel = kernel / np.sum(kernel)
        kernel_size = 2 * self.R + 1
        padded = np.zeros((self.size, self.size), dtype=np.float32)
        start = (self.size - kernel_size) // 2
        padded[start:start+kernel_size, start:start+kernel_size] = kernel
        self.kernel_fft = np.fft.rfft2(np.fft.fftshift(padded))[:, :self.size//2 + 1]
    
    def _growth_function(self, x):
        return np.maximum(0, 1 - (x - self.T)**2 / self.sigma) * 2 - 1
    
    def draw(self):
        fft_grid = np.fft.rfft2(self.grid, axes=(0, 1))
        potential = np.fft.irfft2(
            fft_grid * self.kernel_fft[..., None], 
            s=(self.size, self.size), 
            axes=(0, 1)
        )
        self.grid += self.dt * self._growth_function(potential)
        np.clip(self.grid, 0, 1, out=self.grid)
        return self.grid[:, :, :3]
    
    def spray(self, x, y):
        #### Spray receives the (x, y) position of the mouse when pressed
        #### Then makes a ball of a certain radius and randomly puts stuff in it
        r = self.brush_radius
        i, j = np.ogrid[-r:r+1, -r:r+1]
        distances = np.sqrt(i**2 + j**2)
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.brush_density)
        x_indices = (x + i) % self.size
        y_indices = (y + j) % self.size
        mask = spray_pattern[..., None] 
        random_values = np.random.uniform(0.5, 1, self.channels)
        self.grid[x_indices, y_indices] = np.where(
            mask, random_values, self.grid[x_indices, y_indices]
        )
        
auto = Automaton()

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)
    `
}

export default saved_automata