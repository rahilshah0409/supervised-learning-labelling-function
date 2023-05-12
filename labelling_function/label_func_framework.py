from labelling_function.mlp import State2EventNet
from labelling_function.mlp_training import eval_model, train_model
import wandb
import random

# TODO: Impelement using functions already implemented in other files
def generate_dataset(dataset_dir_path, use_velocities):
    # Generate training data without labels (images and metadata)

    # Segment the images with Segment Anything

    # Run algorithm to get the events for each state generated. num_events should be changed here based on empirical analysis on the training data and the labelling done on it
    return None

# This function should analyse the following:
# How much of each label appears
# The size
# Accuracy of labelling- how does accuracy of labelling translate when we have potentially irrelevant events- would we calculate precision, recall and F1 score?
# Anything else?
def analyse_dataset(dataset):
    print("We are now going to analyse the dataset")

def run_labelling_func_framework():
    train_data_path = ""
    test_data_path = ""
    # Determines whether or not the balls are frozen
    use_velocities = False
    num_events = 0

    # Generate test data without labels

    # Segment the images with Segment Anything

    # Run algorithm to label dataset

    input_size = 52 if use_velocities else 28
    num_layers = 0
    num_neurons = 2
    learning_rate = 0.01
    num_train_epochs = 500
    train_batch_size = 32
    test_batch_size = train_batch_size
    labelling_function = State2EventNet(input_size, num_events, num_layers, num_neurons)

    # Once made, extract datasets from relevant place
    train_data = generate_dataset(train_data_path, use_velocities=use_velocities)
    test_data = generate_dataset(test_data_path)
    # TODO: Need to check quality of training and test dataset created by specified metrics
    
    train_model(labelling_function, learning_rate, num_train_epochs, train_data, train_batch_size)

    eval_model(labelling_function, test_data, test_batch_size)

    return labelling_function

if __name__ == "__main__":
    # Sets up weights and biases for monitoring progress. Can I also use it for showing analysis of dataset
    wandb.init(
        project="labelling-function-learning",
        config={
            "learning_rate": 0.01,
            "epochs": 50,
            "num_layers": 6,
            "num_neurons": 64
        }
    )
    labelling_function = run_labelling_func_framework()
