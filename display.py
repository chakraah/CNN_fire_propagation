import numpy as np
import matplotlib.pyplot as plt

def state_to_rgb(fire_state, fuel_state):

    # Create fire layer (red channel)
    fire_layer = np.copy(fire_state)
    fire_layer[fire_layer > 0] = 255

    # Create fuel layer (green channel)
    fuel_layer = np.copy(fuel_state)
    fuel_layer[fire_layer > 0] = 0  # Remove fuel where fire is burning

    # Combine layers into an RGB image
    simulation_frame = np.stack([fire_layer, fuel_layer, np.zeros_like(fire_layer)], axis=-1)

    return simulation_frame

# Visualising the simulation
def display_fire_spread(fire_history_label, fuel_history_label, fire_history_output, fuel_history_output):

    num_timesteps = len(fire_history_label)
    fig, axes = plt.subplots(1, 3)

     # Create initial empty images for updating
    im_true = axes[0].imshow(np.zeros_like(fire_history_label[0]), animated=True)
    im_pred = axes[1].imshow(np.zeros_like(fire_history_label[0]), animated=True)
    im_diff = axes[2].imshow(np.zeros_like(fire_history_label[0]), animated=True)

    axes[0].set_title('True simulation')
    axes[1].set_title('Predicted simulation')
    axes[2].set_title('Difference')
    
    for timestep in range(num_timesteps):
        # Combine layers into an RGB image for true simulation
        simulation_frame_true = state_to_rgb(fire_history_label[timestep], fuel_history_label[timestep])
        im_true.set_data(simulation_frame_true.astype(np.uint8))
        
        # Combine layers into an RGB image for predicted simulation
        simulation_frame_pred = state_to_rgb(fire_history_output[timestep], fuel_history_output[timestep])
        im_pred.set_data(simulation_frame_pred.astype(np.uint8))

        # Combine layers into an RGB image for difference simulation
        simulation_frame_diff = state_to_rgb(np.abs(fire_history_output[timestep]-fire_history_label[timestep]), 
                                             np.zeros_like(fire_history_output[timestep]))
        im_diff.set_data(simulation_frame_diff.astype(np.uint8))

        plt.pause(0.01)

    plt.show()
