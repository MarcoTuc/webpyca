import torch
import torch.nn.functional as F

class Lenia:
    def __init__(self, size, device=None):
        # Allow specifying device or auto-detect
        self.device = device if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        self.size = size
        
        # Initialize grid with random values on the specified device
        random_uniform = torch.rand((size, size), device=self.device)
        random_mask = torch.bernoulli(torch.ones((size, size), device=self.device) * 0.1)
        self.grid = random_uniform * random_mask
        
        # Parameters
        self.radius = 30
        self.density = 0.6
        
        self.dt = 0.02
        self.R = 14
        self.T = 0.18
        self.sigma = 0.015
        
        # Pre-compute kernel
        self.kernel = self._create_kernel()
        self.kernel_fft = torch.fft.rfft2(self._pad_kernel())
    
    def _create_kernel(self):
        x = torch.linspace(-self.R, self.R, 2*self.R + 1, device=self.device)
        y = x.unsqueeze(1)
        x = x.unsqueeze(0)
        
        dists = torch.sqrt(x**2 + y**2) / self.R
        kernel = torch.exp(-((dists - 0.5) ** 2) / (2 * 0.15**2))
        kernel[dists > 1] = 0
        return kernel / kernel.sum()
    
    def _pad_kernel(self):
        kernel_size = 2 * self.R + 1
        padded = torch.zeros((self.size, self.size), device=self.device)
        start = (self.size - kernel_size) // 2
        padded[start:start+kernel_size, start:start+kernel_size] = self.kernel
        return torch.fft.fftshift(padded)
    
    def _growth_function(self, x):
        return torch.exp(-((x - self.T) ** 2) / (2 * self.sigma**2)) * 2 - 1
    
    def draw(self):
        # Use real-to-complex FFT for better performance
        fft_grid = torch.fft.rfft2(self.grid)
        potential = torch.fft.irfft2(fft_grid * self.kernel_fft)
        growth = self._growth_function(potential)
        
        # In-place operations for better memory efficiency
        self.grid.add_(self.dt * growth).clamp_(0, 1)
        return self.grid
    
    def spray(self, x, y):
        r = self.radius
        
        # Create coordinate grids
        i, j = torch.meshgrid(
            torch.arange(-r, r+1, device=self.device),
            torch.arange(-r, r+1, device=self.device),
            indexing='ij'
        )
        
        # Calculate distances and create spray pattern
        distances = torch.sqrt(i**2 + j**2)
        random_values = torch.rand((2*r+1, 2*r+1), device=self.device)
        spray_pattern = (distances <= r) & (random_values < self.density)
        
        # Apply spray pattern using modulo for wrapping
        for di in range(-r, r+1):
            for dj in range(-r, r+1):
                if spray_pattern[di+r, dj+r]:
                    xi, yj = (x + di) % self.size, (y + dj) % self.size
                    self.grid[xi, yj] = 1.0
    
    def to(self, device):
        """Move computation to the specified device"""
        self.device = torch.device(device)
        self.grid = self.grid.to(self.device)
        self.kernel = self.kernel.to(self.device)
        self.kernel_fft = self.kernel_fft.to(self.device)
        return self
        
    def clear(self):
        """Clear the grid"""
        self.grid.zero_()
        return self


# import numpy as np 

# class Lenia:

#     def __init__(self, size):
#         self.size = size
#         self.grid = np.random.uniform(
#             0, 1, 
#             size=(size, size)
#         ) * np.random.choice(
#             [0, 1], 
#             size=(size, size), 
#             p=[0.9, 0.1]
#         )
#         self.radius = 30
#         self.density = 0.6
        
#         self.dt = 0.02
#         self.R = 14
#         self.T = 0.18
#         self.sigma = 0.015
        
#         self.kernel = self._create_kernel()
#         self.kernel_fft = np.fft.fft2(self._pad_kernel())
    
#     def _create_kernel(self):
#         x = np.linspace(-self.R, self.R, 2*self.R + 1)
#         dists = np.sqrt(x[:, None]**2 + x[None, :]**2) / self.R
#         kernel = np.exp(-((dists - 0.5) ** 2) / (2 * 0.15**2))
#         kernel[dists > 1] = 0
#         return kernel / np.sum(kernel)
    
#     def _pad_kernel(self):
#         kernel_size = 2 * self.R + 1
#         padded = np.zeros((self.size, self.size))
#         start = (self.size - kernel_size) // 2
#         padded[start:start+kernel_size, start:start+kernel_size] = self.kernel
#         return np.fft.fftshift(padded)
    
#     def _growth_function(self, x):
#         return np.exp(-((x - self.T) ** 2) / (2 * self.sigma**2)) * 2 - 1
    
#     def draw(self):
#         fft_grid = np.fft.fft2(self.grid)
#         potential = np.real(np.fft.ifft2(fft_grid * self.kernel_fft))
#         self.grid = np.clip(
#             self.grid + self.dt * self._growth_function(potential), 0, 1
#         )
#         return self.grid
    
#     def spray(self, x, y):
#         #### Receives an (x, y) position from the mouse
#         #### and draws a ball around it when pressed  
#         r = self.radius
#         i, j = np.ogrid[-r:r+1, -r:r+1]
#         distances = np.sqrt(i**2 + j**2)
#         spray_pattern = (distances <= r) & (np.random.random(
#                                             (2*r + 1, 2*r + 1)) < self.density)
#         x_indices = (x + i) % self.size
#         y_indices = (y + j) % self.size
#         self.grid[x_indices, y_indices] = np.where(
#             spray_pattern, 1, self.grid[x_indices, y_indices]
#         )