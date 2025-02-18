import numpy as np
import scipy.signal

class WildfireSimulation:
      
    def __init__(self, grid_size, tree_density, timesteps, burn_rate, ignition_threshold, ignition_points, wind_direction):
        
        self.grid_size = grid_size
        self.tree_density = tree_density
        self.burn_rate = burn_rate
        self.timesteps = timesteps
        self.ignition_threshold = ignition_threshold
        self.ignition_points = ignition_points

         # Wind parameters (None means no wind)
        self.wind_direction = wind_direction  # (angle in radians) 0 = right, pi/2 = up, etc.
        
        self.initial_fire_state = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.ignite_at_points()
        self.fuel_map = self.generate_fuel_map()

    ## | ---------------- Function: generating a fuel map for a forest fire simulation --------------- |
    
    def generate_fuel_map(self):

        np.random.seed(42)  # Ensure reproducibility
        
        # Initialize grid with no fuel
        fuel_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32) 
        
        # Calculate number of tree-covered cells
        num_trees = int(self.grid_size**2 * self.tree_density)  
        
        # Generate random tree positions
        tree_positions = np.random.randint(0, self.grid_size, (num_trees, 2))
        
        # Assign random fuel values (0-255) to tree positions
        fuel_map[tree_positions[:, 0], tree_positions[:, 1]] = np.random.randint(0, 255, num_trees)

        # Add a no-fuel area 
        # pos = np.random.randint(50, self.grid_size-50)

        # fuel_map[pos:pos+30, pos-10:pos] = 0
        # fuel_map[pos+20:pos+30, pos-20:pos] = 0
        
        return fuel_map
    
    def create_wind_kernel(self):

        wind_kernel = np.zeros((3, 3), dtype=np.float32)
        
        if self.wind_direction == np.pi/2:    # Up/North (pi/2)
            wind_kernel[0, :] = 1
        elif self.wind_direction == -np.pi/2:  # Down/South (3pi/2)
            wind_kernel[2, :] = 1
        elif self.wind_direction == 0:  # Right/East (0)
            wind_kernel[:, 2] = 1
        elif self.wind_direction == np.pi:  # Left/West (pi)
            wind_kernel[:, 0] = 1
            
        return wind_kernel

    ## | ---------------- Function: ignition of the fire at the center of the grid --------------- |

    def ignite_at_points(self):

        for point in self.ignition_points:
            x, y = point
            self.initial_fire_state[x - 2:x + 2, y - 2:y + 2] = 1

    ## | ---------------- Function: wildfire simulation --------------- |
    
    def run_simulation(self):
        
        # Initialize state and fuel arrays
        fire_state = np.copy(self.initial_fire_state) 
        fuel = np.copy(self.fuel_map)
        
        # Storage for simulation history
        fire_history = np.zeros((self.timesteps, *fire_state.shape), dtype=np.float32)
        fuel_history = np.zeros((self.timesteps, *fire_state.shape), dtype=np.float32)

        # Get the wind kernel
        wind_kernel = self.create_wind_kernel()

        for t in range(self.timesteps):

            # Apply wind effect: Convolve fire state with wind kernel
            wind_influence = scipy.signal.convolve2d(fire_state, wind_kernel, mode='same', boundary='fill', fillvalue=0)

            # Update fire propagation with wind influence
            new_ignitions = (wind_influence > self.ignition_threshold) * fuel
            
            fire_state += new_ignitions
            fire_state[fire_state>0] = 1
            
            # Identify burning and burned-out cells
            burning_cells = fire_state > 0
            burned_out_cells = fuel < self.burn_rate
            
            # Reduce fuel and update fire state
            fuel[burning_cells] -= self.burn_rate
            fire_state[burned_out_cells] = 0  # Fire extinguishes if fuel is depleted
            fuel[burned_out_cells] = 0 
            
            # Store simulation history
            fire_history[t] = fire_state
            fuel_history[t] = fuel
        
        return fire_history, fuel_history

