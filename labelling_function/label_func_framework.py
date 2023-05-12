from labelling_function.mlp import State2EventNet
from labelling_function.mlp_training import eval_model, train_model

# TODO: Impelement using functions already implemented in other files
def generate_dataset(dataset_dir_path):
    # Generate training data without labels (images and metadata)

    # Segment the images with Segment Anything

    # Run algorithm to get the events for each state generated. num_events should be changed here based on empirical analysis on the training data and the labelling done on it
    return None

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
    train_data = generate_dataset(train_data_path)
    test_data = generate_dataset(test_data_path)
    # TODO: Need to check quality of training and test dataset created by specified metrics

    
    train_model(labelling_function, learning_rate, num_train_epochs, train_data, train_batch_size)

    eval_model(labelling_function, test_data, test_batch_size)

    return labelling_function

if __name__ == "__main__":
    labelling_function = run_labelling_func_framework()
