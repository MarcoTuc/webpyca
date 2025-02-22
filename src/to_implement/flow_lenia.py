import numpy as np 
import scipy
import scipy.signal

#### FLOW LENIA
#### marco.tuccio95@gmail.com

#### click 'Run' to initiate the simulation
#### you can control target FPS below 'Run'

class Automaton:
    def __init__(self, size):
        self.size = size
        
        # System parameters - optimized for speed
        self.dt = 0.2
        self.dd = 3        # Reduced flow distance for speed
        self.sigma = 0.65
        self.R = 13
        self.nb_k = 3
        
        # Pre-compute constants
        self.m = np.random.uniform(0.2, 0.3, self.nb_k)
        self.s = np.random.uniform(0.01, 0.05, self.nb_k)
        self.h = np.random.uniform(0.1, 0.3, self.nb_k)
        
        # Initialize grid with sparse random values
        self.grid = np.zeros((size, size))
        center = size // 2
        radius = 20
        self.grid[center-radius:center+radius, center-radius:center+radius] = \
            np.random.rand(2*radius, 2*radius) * 0.5
        
        # Pre-compute kernels and FFT plans
        self._initialize_kernels()
        
        # Pre-compute flow weights
        self._initialize_flow_weights()
        
        # Brush parameters
        self.radius = 24
        self.density = 0.6

    def _initialize_kernels(self):
        # Create kernels more efficiently
        x = np.linspace(-self.R, self.R, 2*self.R + 1)
        xx, yy = np.meshgrid(x, x)
        dists = np.sqrt(xx**2 + yy**2) / self.R
        
        # Pre-allocate kernel array
        self.kernels = np.zeros((self.size, self.size, self.nb_k), dtype=np.float32)
        kernel_size = 2 * self.R + 1
        start = (self.size - kernel_size) // 2
        
        # Create base kernel once
        base_kernel = np.exp(-((dists - 0.5)**2) / (2 * 0.15**2))
        base_kernel[dists > 1] = 0
        base_kernel /= base_kernel.sum()
        
        # Replicate with variations
        for k in range(self.nb_k):
            self.kernels[start:start+kernel_size, start:start+kernel_size, k] = \
                base_kernel * (0.9 + 0.2 * np.random.random())
        
        # Pre-compute FFT of kernels
        self.kernel_fft = np.fft.fft2(np.fft.fftshift(self.kernels, axes=(0, 1)), axes=(0, 1))

    def _initialize_flow_weights(self):
        # Pre-compute flow weights matrix
        dd_range = np.arange(-self.dd, self.dd + 1)
        xx, yy = np.meshgrid(dd_range, dd_range)
        dists = np.sqrt(xx**2 + yy**2)
        self.flow_weights = np.exp(-dists**2 / (2 * self.sigma**2))
        self.flow_weights /= self.flow_weights.sum()
        
        # Pre-compute Sobel kernels
        self.sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 8
        self.sobel_y = self.sobel_x.T

    def _growth(self, U):
        # Vectorized growth function
        return np.sum([
            self.h[k] * (np.exp(-((U - self.m[k])**2) / (2 * self.s[k]**2)) * 2 - 1)
            for k in range(self.nb_k)
        ], axis=0)

    def _sobel(self, A):
        # Optimized Sobel using pre-computed kernels
        gx = np.zeros_like(A)
        gy = np.zeros_like(A)
        
        gx = scipy.signal.correlate2d(A, self.sobel_x, mode='same', boundary='wrap')
        gy = scipy.signal.correlate2d(A, self.sobel_y, mode='same', boundary='wrap')
        
        return np.stack([gy, gx], axis=-1)

    def draw(self):
        # Calculate potential field using FFT
        fft_grid = np.fft.fft2(self.grid[..., None], axes=(0, 1))
        U = np.real(np.fft.ifft2(fft_grid * self.kernel_fft, axes=(0, 1)))
        
        # Calculate growth
        G = self._growth(U[..., 0])
        
        # Calculate flow field
        F = self._sobel(G)
        C_grad = self._sobel(self.grid)
        
        # Combine growth and flow
        alpha = np.clip((self.grid[..., None] / 2)**2, 0, 1)
        F = F * (1 - alpha) - C_grad * alpha
        F = np.clip(F, -self.dd + self.sigma, self.dd - self.sigma)
        
        # Update using pre-computed flow weights
        new_grid = np.zeros_like(self.grid)
        pad_width = self.dd
        padded = np.pad(self.grid, pad_width, mode='wrap')
        
        # Use vectorized operation instead of loops
        for i in range(2*self.dd + 1):
            for j in range(2*self.dd + 1):
                new_grid += (padded[i:i+self.size, j:j+self.size] * 
                           self.flow_weights[i, j])
        
        # Update state
        total_mass = new_grid.sum() + 1e-10
        self.grid = np.clip(
            new_grid * self.grid.sum() / total_mass + G * self.dt,
            0, 1
        )
        return self.grid
    
    def spray(self, x, y):
        # Optimized spray function using vectorized operations
        r = self.radius
        y_coords, x_coords = np.ogrid[-r:r+1, -r:r+1]
        distances = np.sqrt(x_coords**2 + y_coords**2)
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.density)
        
        x_indices = (x + np.arange(-r, r+1)) % self.size
        y_indices = (y + np.arange(-r, r+1)) % self.size
        xx, yy = np.meshgrid(x_indices, y_indices)
        
        self.grid[xx, yy] = np.where(spray_pattern, np.random.uniform(0.5, 1.0), 
                                    self.grid[xx, yy])

auto = Automaton(300)

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)