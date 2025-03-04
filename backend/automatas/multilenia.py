import numpy as np

world_size = 150
number_hidden_channels = 5

brush_radius = 50
brush_density = 0.2

R = 15
dt = 0.2
power_T = 0.1
power_sigma = 0.3

class MultiLenia:
    
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