import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from fire_propagation_cnn import FirePropagationCNN
from fire_propagation_cnn import FirePropagationDataset
from fire_propagation_cnn import ModelTrainer

from fire_propagation_model.fire_propagation_model import WildfireSimulation

from display import display_fire_spread


def create_fire_data(grid_size, num_samples, wind_direction):
    # Generates fire propagation data for training/testing
    ignition_point = [(np.random.randint(1,grid_size-1), np.random.randint(1,grid_size-1))]
    wildfire_simulation = WildfireSimulation(grid_size, 0.9, num_samples, 5, 0.2, ignition_point, wind_direction)
        
    fire_data, fuel_history = wildfire_simulation.run_simulation()

    inputs, targets = [], []

    for i in range(len(fire_data) - 1):
        inputs.append((fire_data[i], fuel_history[i]/255))
        targets.append((fire_data[i+1], fuel_history[i+1]/255))

    return inputs, targets, fuel_history


# Main Execution
if __name__ == "__main__":

    # Environment parameters
    GRID_SIZE = 200
    WIND_DIRECTION = 0  # Wind blowing towards the right
    NUM_SAMPLES = 150
    NUM_SCENARIOS = 20

    # Define model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "fire_propagation_model_200.pth")

    # Initialize model
    model = FirePropagationCNN()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)

    if not os.path.exists(model_path):

        # Training parameters
        BATCH_SIZE = 16
        LEARNING_RATE = 1e-4
        NUM_EPOCHS = 200

        # Initialize loss function, and optimizer
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

        for scenario_idx in range(NUM_SCENARIOS):

            # Generate dataset
            fire_inputs, fire_targets, x = create_fire_data(GRID_SIZE, NUM_SAMPLES, WIND_DIRECTION)
            dataset = FirePropagationDataset(fire_inputs, fire_targets)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Train the model
            trainer = ModelTrainer(model, dataloader, loss_function, optimizer, NUM_EPOCHS, DEVICE)
            trainer.train()

        # Save trained model
        print("Training complete for all scenarios, saving model...")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")

    else:
        print(f"Model found, loading...")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Evaluation
    fire_state_history_labels = []
    fuel_map_history_labels = []
    fire_state_history_predictions = []
    fuel_map_history_predictions = []

    fire_inputs, fire_targets, fuel_history = create_fire_data(GRID_SIZE, NUM_SAMPLES, WIND_DIRECTION)
    dataset = FirePropagationDataset(fire_inputs, fire_targets)

    sample_input, sample_label = dataset[0] 
    sample_input = sample_input.unsqueeze(0).to(DEVICE)

    for timestep in range(NUM_SAMPLES - 2):

        model.eval()

        with torch.no_grad():
            sample_output = model(sample_input)

        sample_output = sample_output[0].cpu().numpy()

        # Extract labels from the ground truth
        fire_state_label = sample_label[0].cpu().numpy()
        fuel_map_label = sample_label[1].cpu().numpy() * 255
        
        # Extract outputs from the model
        fire_state_prediction = sample_output[0]
        fuel_map_prediction = sample_output[1] * 255

        # Count the number of ones in the fire_state_label array
        num_ones = np.sum(fire_state_label == 1)

        if num_ones > 0:

            # Flatten the fire_state_prediction and sort the values
            flattened_predictions = fire_state_prediction.flatten()

            # Sort the predictions to determine which values should be set to 1
            top_indices = np.sort(flattened_predictions)[-num_ones:]  # Exclude last 'num_ones' smallest values

            # Find the minimum value in the top indices to create a threshold
            threshold = np.min(top_indices)

            # Update the prediction array based on the threshold
            fire_state_prediction[fire_state_prediction >= threshold] = 1
            fire_state_prediction[fire_state_prediction < threshold] = 0
        else:
            fire_state_prediction[fire_state_prediction < 0.1] = 0

        # Store results
        fire_state_history_labels.append(fire_state_label)
        fuel_map_history_labels.append(fuel_map_label.astype(int))
        fire_state_history_predictions.append(fire_state_prediction)
        fuel_map_history_predictions.append(fuel_map_prediction.astype(int))

        # Prepare next input and target
        sample_input = torch.stack([torch.tensor(fire_state_prediction), torch.tensor(fuel_map_prediction / 255)]).unsqueeze(0).to(DEVICE)
        _, sample_label = dataset[timestep+1] 

    # Visualize simulation results
    display_fire_spread(fire_state_history_labels, fuel_map_history_labels, fire_state_history_predictions, fuel_map_history_predictions)
