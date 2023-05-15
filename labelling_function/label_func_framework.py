import sys
sys.path.insert(1, "../")
from image_segmentation.dataset_labelling_pipeline import generate_and_save_masks_for_eps, generate_event_labels_from_masks
from image_generation.generate_imgs import run_rand_policy_and_save_traces
from labelling_model import State2EventNet
from model_training import eval_model, train_model
import wandb
import gym
import pickle

def generate_unlabelled_images(use_velocities, dataset_dir_path, img_base_fname):
    env = gym.make("gym_subgoal_automata:WaterWorldDummy-v0",
                   params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 200})
    random_seed = None
    num_episodes = 5

    # Generate training data without labels (images and metadata)
    run_rand_policy_and_save_traces(env, num_episodes, dataset_dir_path, img_base_fname, random_seed)

def label_dataset(img_dir_path, img_base_fname):
    # Segment the images with Segment Anything
    trace_data_filename = "traces_data.pkl"
    with open(img_dir_path + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    
    # The following lines need to be run on a lab machine
    sam_checkpoint = "/vol/bitbucket/ras19/fyp/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    filtered_masks_pkl_name = "masks.pkl"
    # masks_for_every_ep = generate_and_save_masks_for_eps(trace_data, img_dir_path, sam_checkpoint, model_type, img_base_fname, filtered_masks_pkl_name)

    # Run algorithm to get the events for each state generated. num_events should be changed here based on empirical analysis on the training data and the labelling done on it
    events_fname = "events.pkl"
    events_for_every_ep, events_observed = generate_event_labels_from_masks(trace_data, img_dir_path, model_type, filtered_masks_pkl_name, img_base_fname, events_fname)
    return img_dir_path, events_observed

# This function should analyse the following:
# How much of each label appears
# The size
# Accuracy of labelling- how does accuracy of labelling translate when we have potentially irrelevant events- would we calculate precision, recall and F1 score?
# Anything else?
def analyse_dataset(dataset):
    print("We are now going to analyse the dataset")

def run_labelling_func_framework():
    # Determines whether or not the balls are frozen
    use_velocities = False
    num_events = 0

    # Generate training data
    train_data_dir = "../dataset2/training/"
    img_base_fname = "step"
    test_data_dir = "../dataset2/test/"
    test_img_base_fname = "test_step"

    # generate_unlabelled_images(use_velocities, train_data_dir, img_base_fname)
    # label_dataset(train_data_dir, img_base_fname)

    # Generate test data
    generate_unlabelled_images(use_velocities, test_data_dir, test_img_base_fname)
    # label_dataset(test_data_dir, test_img_base_fname)

    # TODO: Need to check quality of training and test dataset created by specified metrics

    input_size = 52 if use_velocities else 28
    num_layers = 0
    num_neurons = 2
    learning_rate = 0.01
    num_train_epochs = 500
    train_batch_size = 32
    test_batch_size = train_batch_size
    labelling_function = State2EventNet(input_size, num_events, num_layers, num_neurons)
    
    # train_model(labelling_function, learning_rate, num_train_epochs, train_data, train_batch_size)

    # eval_model(labelling_function, test_data, test_batch_size)

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
