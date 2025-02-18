import numpy as np
import matplotlib.pyplot as plt

from fire_propagation_model import WildfireSimulation

# Visualising the simulation
def display_fire_propagation(fire_history, fuel_history):

    num_timesteps = len(fire_history)
    plt.figure(figsize=(8, 8))
    
    for timestep in range(num_timesteps):
        # Create fire layer (red channel)
        fire_layer = np.copy(fire_history[timestep])
        fire_layer[fire_layer > 0] = 255
        
        # Create fuel layer (green channel)
        fuel_layer = np.copy(fuel_history[timestep])
        fuel_layer[fire_layer > 0] = 0  # Remove fuel where fire is burning
        
        # Combine layers into an RGB image
        simulation_frame = np.stack([fire_layer, fuel_layer, np.zeros_like(fire_layer)], axis=-1)

        # Display the simulation frame
        plt.imshow(simulation_frame.astype(np.uint8))
        plt.title(f'Timestep {timestep+1}')
        plt.axis('off')
        plt.pause(0.001)
    
    plt.show()

# Main execution of the Wildfire Simulation
if __name__ == "__main__":

    # Environment parameters
    GRID_SIZE = 200
    TREE_DENSITY = 0.9
    NUM_STEPS = 200
    BURN_RATE = 5
    IGNITION_THRESHOLD = 0.005
    WIND_DIRECTION = 0

    IGNITION_POINT = [(100,100), (150,5)]

    wildfire_simulation = WildfireSimulation(GRID_SIZE, TREE_DENSITY, NUM_STEPS, BURN_RATE, IGNITION_THRESHOLD, IGNITION_POINT, WIND_DIRECTION)

    # Run the simulation and get the fire and fuel history
    fire_history, fuel_history = wildfire_simulation.run_simulation()

    # Visualize the results of the simulation
    display_fire_propagation(fire_history, fuel_history)
