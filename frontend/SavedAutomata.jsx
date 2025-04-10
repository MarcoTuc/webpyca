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

auto = Automaton(300)

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
    `,

    leniaxplorer: 
    `



    # ONE CHANNEL LENIA IMPLEMENTATION
    # Based on BatchLenia with parameters from read_data.py
    
    import numpy as np
from scipy.fft import fft2, ifft2


class LeniaParams:
    
    """NumPy version of LeniaParams to store and manage Lenia parameters."""
    
    def __init__(self, batch_size=1, k_size=25, channels=1, param_dict=None):
        self.batch_size = batch_size
        self.k_size = k_size
        self.channels = channels
        
        if param_dict is not None:
            self.param_dict = param_dict
            self.mu = np.array(param_dict['mu'])
            self.sigma = np.array(param_dict['sigma'])
            self.beta = np.array(param_dict['beta'])
            self.mu_k = np.array(param_dict['mu_k'])
            self.sigma_k = np.array(param_dict['sigma_k'])
            self.weights = np.array(param_dict['weights'])
            self.k_size = param_dict.get('k_size', k_size)
            self.batch_size = self.weights.shape[0]
        else:
            # Initialize with default values
            self.mu = np.array([[[0.1]]])
            self.sigma = np.array([[[0.015]]])
            self.beta = np.array([[[[1.0]]]])
            self.mu_k = np.array([[[[0.5]]]])
            self.sigma_k = np.array([[[[0.15]]]])
            self.weights = np.array([[[1.0]]])
            self.param_dict = {
                'k_size': self.k_size,
                'mu': self.mu,
                'sigma': self.sigma,
                'beta': self.beta,
                'mu_k': self.mu_k,
                'sigma_k': self.sigma_k,
                'weights': self.weights
            }
    
    def __getitem__(self, key):
        return self.param_dict[key]


class Automaton:
    """Base automaton class."""
    def __init__(self, size):
        self.h, self.w = size
        self._worldmap = None
        self.worldsurface = None
    
    @property
    def worldmap(self):
        return self._worldmap
    
    @worldmap.setter
    def worldmap(self, value):
        self._worldmap = value
        

class MultiLeniaNumPy(Automaton):
    """
    Multi-channel Lenia automaton implemented in NumPy.
    A multi-colored GoL-inspired continuous automaton. Originally introduced by Bert Chan.
    """
    def __init__(
            self, 
            size, 
            batch=1, 
            dt=0.1, 
            num_channels=3, 
            params=None, 
            param_path=None, 
            seeds=None,
            ):
        """
        Initializes automaton.  

        Args:
            size: (H,W) of ints, size of the automaton
            batch: int, batch size for parallel simulations
            dt: float, time-step used when computing the evolution
            num_channels: int, number of channels (C)
            params: LeniaParams class or dict of parameters
            param_path: str, path to folder containing saved parameters
            seeds: a list of seeds, one per batch size so len(seeds) == batch is True
        """
        super().__init__(size=size)


        self.batch = batch
        self.C = num_channels

        if params is None:
            self.params = LeniaParams(batch_size=self.batch, k_size=25, channels=self.C)
        elif isinstance(params, dict):
            self.params = LeniaParams(param_dict=params)
        elif isinstance(params, LeniaParams):
            self.params = params
        else: raise TypeError(f"parameters should be of type LeniaParams or Dict and not {type(params)}")
    
        ############################################# POLYGON INITIALIZER
        #TODO remove this and make it API dependent
        kernel_folder = "_".join([str(s) for s in self.params["mu_k"][0,0,0]])+"_"+"_".join([str(s) for s in self.params["sigma_k"][0,0,0]])
        self.kernel_path = "unif_random_voronoi/" + kernel_folder
        # Load polygons for initialization
        try:
            with open(f'utils/polygons{self.h}.pickle', 'rb') as handle:
                self.polygons = pickle.load(handle)
        except:
            print("polygons for this array size not generated yet")
            self.polygons = None
        # self.data_path = f"unif_random_voronoi/{kernel_folder}/data/{self.g_mu}_{self.g_sig}.pickle"

        self.seeds = seeds
        if seeds is None:
            self.seeds = [np.random.randint(2**32) for _ in range(self.batch)]        
        if not len(seeds) == self.batch:
            raise ValueError("number of seeds does not match batch size, reinitializing seeds")
        
        # Create a random initial state
        self.state = np.random.uniform(size=(self.batch, self.C, self.h, self.w))
        self.dt = dt
 
        self.g_mu = np.round(params["mu"].item(), 4)
        self.g_sig = np.round(params["sigma"].item(), 4)
        self.k_size = self.params['k_size']
        # Compute normalized weights
        self.normal_weights = self.norm_weights(self.params.weights)
        # Compute kernel and its FFT
        self.kernel = self.compute_kernel()
        self.fft_kernel = self.kernel_to_fft(self.kernel)       

        ii, jj = np.meshgrid(np.arange(0, self.w), np.arange(0, self.h), indexing='ij')
        # Stack and reshape coordinates
        coords = np.stack([np.reshape(ii, (-1,)), np.reshape(jj, (-1,))], axis=-1)
        coords = coords.astype(np.float32)

        # Reshape to (array_size^2, 2, 1)
        self.coords = np.reshape(coords, (self.w*self.h, 2, 1))

        # For loading and saving parameters
        self.saved_param_path = param_path
        if self.saved_param_path is not None:
            self.param_files = [file for file in os.listdir(self.saved_param_path) if file.endswith('.pt')]
            self.num_par = len(self.param_files)
            if self.num_par > 0:
                self.cur_par = random.randint(0, self.num_par-1)
            else:
                self.cur_par = None

        self.to_save_param_path = 'SavedParameters/Lenia'

    #TODO compatibilize with pyodide parameter loading methods I will make 
    def update_params(self, params, k_size_override=None):
        """
        Updates parameters of the automaton.
        Args:
            params: LeniaParams object
            k_size_override: int, override the kernel size of params
        """
        if k_size_override is not None:
            self.k_size = k_size_override
            if self.k_size % 2 == 0:
                self.k_size += 1
                print(f'Increased even kernel size to {self.k_size} to be odd')
            params.k_size = self.k_size
        
        self.params = LeniaParams(param_dict=params.param_dict)
        self.batch = self.params.batch_size
        
        # Update derived parameters
        self.normal_weights = self.norm_weights(self.params.weights)
        self.kernel = self.compute_kernel()
        self.fft_kernel = self.kernel_to_fft(self.kernel)

    @staticmethod
    def norm_weights(weights):
        """
        Normalizes the relative weight sum of the growth functions.
        Args:
            weights: (B,C,C) array of weights 
        Returns:
            (B,C,C) array of normalized weights
        """
        # Sum weights along the first dimension
        sum_weights = np.sum(weights, axis=1, keepdims=True)  # (B,1,C)
        # Normalize weights, avoiding division by zero
        return np.where(sum_weights > 1e-6, weights / sum_weights, 0)

    def set_init_voronoi_batch(self, polygon_size=60, init_polygon_index=0):
        """
        Initialize state using Voronoi polygons.
        
        Args:
            polygon_size: int, size of polygons
            init_polygon_index: int, starting index for polygons
        """

        # Create empty numpy array for states
        states_np = np.empty((self.batch, self.C, self.h, self.w))
        
        for i, seed in enumerate(self.seeds):
            polygon_index = init_polygon_index + i
            mask = self.polygons[polygon_size][polygon_index % 1024]
            mask = load_pattern(mask.reshape(1, *mask.shape), [self.h, self.w]).reshape(self.h, self.w)

            np.random.seed(seed)
            
            # Generate random state and apply mask
            rand_np = np.random.rand(1, self.C, self.h, self.w)
            pattern = np.asarray(rand_np * mask)
            states_np[i] = pattern[0]
        
        # Update state
        self.state = states_np


    def kernel_slice(self, r):
        """
        Given a distance matrix r, computes the kernel of the automaton.
        
        Args:
            r: (k_size,k_size) array, value of the radius for each pixel of the kernel
            
        Returns:
            (B,C,C,k_size,k_size) array of kernel values
        """
        # Expand radius to match expected kernel shape
        r = r[None, None, None, None, :, :]  # (1,1,1,1,k_size,k_size)
        
        # Get number of cores
        num_cores = self.params.mu_k.shape[3]
        
        # Expand r to match batched parameters
        r = np.broadcast_to(r, (self.batch, self.C, self.C, num_cores, self.k_size, self.k_size))
        
        # Reshape parameters for broadcasting
        mu_k = self.params.mu_k[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)
        sigma_k = self.params.sigma_k[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)
        beta = self.params.beta[:, :, :, :, None, None]  # (B,C,C,#cores,1,1)
        
        # Compute kernel
        K = np.exp(-((r - mu_k) / sigma_k)**2 / 2)  # (B,C,C,#cores,k_size,k_size)
        
        # Sum over cores with respective heights
        K = np.sum(beta * K, axis=3)  # (B,C,C,k_size,k_size)
        
        return K

    def compute_kernel(self):
        """
        Computes the kernel given the current parameters.
        
        Returns:
            (B,C,C,k_size,k_size) array of kernel values
        """
        # Create coordinate grid
        xyrange = np.linspace(-1, 1, self.k_size)
        x, y = np.meshgrid(xyrange, xyrange, indexing='xy')
        
        # Compute radius values
        r = np.sqrt(x**2 + y**2)
        
        # Compute kernel
        K = self.kernel_slice(r)  # (B,C,C,k_size,k_size)
        
        # Normalize kernel
        summed = np.sum(K, axis=(-1, -2), keepdims=True)  # (B,C,C,1,1)
        summed = np.where(summed < 1e-6, 1.0, summed)  # Avoid division by zero
        K = K / summed
        
        return K

    def kernel_to_fft(self, K):
        """
        Computes the Fourier transform of the kernel.
        
        Args:
            K: (B,C,C,k_size,k_size) array, the kernel
            
        Returns:
            (B,C,C,h,w) array, the FFT of the kernel
        """
        # Create padded kernel
        padded_K = np.zeros((self.batch, self.C, self.C, self.h, self.w))
        
        # Place kernel in the center
        k_h, k_w = self.k_size, self.k_size
        start_h, start_w = self.h // 2 - k_h // 2, self.w // 2 - k_w // 2
        
        # Update padded kernel with actual kernel values
        padded_K[:, :, :, start_h:start_h+k_h, start_w:start_w+k_w] = K
        
        # Shift for FFT
        padded_K = np.roll(padded_K, [-self.h // 2, -self.w // 2], axis=(-2, -1))
        
        # Compute FFT
        return fft2(padded_K)

    def growth(self, u):
        """
        Computes the growth function applied to concentrations u.
        
        Args:
            u: (B,C,C,h,w) array of concentrations
            
        Returns:
            (B,C,C,h,w) array of growth values
        """
        # Reshape parameters for broadcasting
        mu = self.params.mu[:, :, :, None, None]  # (B,C,C,1,1)
        sigma = self.params.sigma[:, :, :, None, None]  # (B,C,C,1,1)
        
        # Broadcast to match u's shape
        mu = np.broadcast_to(mu, u.shape)
        sigma = np.broadcast_to(sigma, u.shape)
        
        # Compute growth function (Gaussian bump)
        return 2 * np.exp(-((u - mu)**2 / (sigma)**2) / 2) - 1

    def get_fftconv(self, state):
        """
        Compute convolution of state with kernel using FFT.
        
        Args:
            state: (B,C,h,w) array, the current state
            
        Returns:
            (B,C,C,h,w) array of convolution results
        """
        # Compute FFT of state
        fft_state = fft2(state)  # (B,C,h,w)
        
        # Reshape for broadcasting with kernel
        fft_state = fft_state[:, :, None, :, :]  # (B,C,1,h,w)
        
        # Multiply in frequency domain
        convolved = fft_state * self.fft_kernel  # (B,C,C,h,w)
        
        # Inverse FFT
        result = ifft2(convolved)  # (B,C,C,h,w)
        
        return np.real(result)

    def step(self):
        """
        Steps the automaton state by one iteration.
        """
        # Compute convolutions
        convs = self.get_fftconv(self.state)  # (B,C,C,h,w)
        
        # Compute growth
        growths = self.growth(convs)  # (B,C,C,h,w)
        
        # Apply weights
        weights = self.normal_weights[:, :, :, None, None]  # (B,C,C,1,1)
        weights = np.broadcast_to(weights, growths.shape)  # (B,C,C,h,w)
        
        # Sum weighted growths
        dx = np.sum(growths * weights, axis=1)  # (B,C,h,w)
        
        # Update state
        self.state = np.clip(self.state + self.dt * dx, 0, 1)  # (B,C,h,w)

    def mass(self):
        """
        Computes average 'mass' of the automaton for each channel.
        
        Returns:
            (B,C) array, mass of each channel
        """
        return np.mean(self.state, axis=(-1, -2))  # (B,C)
    
    def get_batch_mass_center(self, array):
        """
        Calculate the center of mass for each batch and channel.
        
        Args:
            array: (B,C,H,W) array, the current state
            
        Returns:
            tuple of (2, B*C) array of center coordinates and (B*C) array of masses
        """
        B, C, H, W = array.shape  # array shape: (B,C,H,W)
        
        # Reshape array to (H*W, 1, B*C)
        A = np.transpose(array, (2, 3, 0, 1)).reshape(H*W, 1, B*C)
        
        # Calculate total mass
        total_mass = np.sum(A, axis=0)[-1]  # (B*C)
        
        # Calculate weighted sum by coordinates
        prod = A * self.coords
        sum_mass = np.sum(prod, axis=0)  # (2, B*C)
        
        # Create a mask for non-zero masses
        mask = (total_mass != 0)
        
        # Normalize by total mass where total mass is not zero
        sum_mass[:, mask] = sum_mass[:, mask] / total_mass[mask]
    
        return sum_mass, total_mass

    def draw(self):
        """
        Draws the RGB worldmap from state.
        """
        assert self.state.shape[0] == 1, "Batch size must be 1 to draw"
        toshow = self.state[0]  # (C,h,w)

        if self.C == 1:
            # Expand to 3 channels
            toshow = np.broadcast_to(toshow, (3, self.h, self.w))
        elif self.C == 2:
            # Add zero channel
            zeros = np.zeros((1, self.h, self.w))
            toshow = np.concatenate([toshow, zeros], axis=0)
        else:
            # Use only first 3 channels
            toshow = toshow[:3, :, :]
    
        self._worldmap = toshow
        
        return toshow


params = {
    'k_size': 27, 
    'mu': np.array([[[0.15]]]), 
    'sigma': np.array([[[0.015]]]), 
    'beta': np.array([[[[1.0]]]]), 
    'mu_k': np.array([[[[0.5]]]]), 
    'sigma_k': np.array([[[[0.15]]]]), 
    'weights': np.array([[[1.0]]])
}


seeds = [4218145047]
polygon_size = 50
array_size = 150

lenia_numpy = MultiLeniaNumPy((array_size, array_size), batch=1, num_channels=1, dt=0.1, params=params, seeds=seeds)

def main():
    lenia_numpy.step()
    world = lenia_numpy.draw()[0]
    return world






        
    `
}

export default saved_automata