import sys
sys.path.insert(1, "../")
from image_segmentation.dataset_labelling_pipeline import generate_and_save_masks_for_eps, generate_event_labels_from_masks
from image_generation.generate_imgs import run_rand_policy_and_save_traces
from labelling_model import State2EventNet
from model_training import eval_model, train_model
import wandb
import gym
import pickle
import numpy as np
import random
import math
import torch
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

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
    filtered_masks_pkl_name_base = "masks"
    # masks_for_every_ep = generate_and_save_masks_for_eps(trace_data, img_dir_path, sam_checkpoint, model_type, img_base_fname, filtered_masks_pkl_name_base)

    # Run algorithm to get the events for each state generated. num_events should be changed here based on empirical analysis on the training data and the labelling done on it
    events_fname = "events_2.pkl"
    events_for_every_ep, events_observed = generate_event_labels_from_masks(trace_data, img_dir_path, model_type, filtered_masks_pkl_name_base, img_base_fname, events_fname)
    return img_dir_path, events_observed

# This function should analyse the following:
# How much of each label appears
# The size
# Accuracy of labelling- how does accuracy of labelling translate when we have potentially irrelevant events- would we calculate precision, recall and F1 score?
# Anything else?
# The dataset passed in as an argument in this function has datapoints in the form of tuples 
def analyse_dataset(dataset, events_captured):
    print("The size of the dataset is {}".format(len(dataset)))
    label_distribution = _get_distribution_of_labels(dataset, events_captured)
    print("The label distribution")
    print(label_distribution)

def _get_distribution_of_labels(dataset, events_captured):
    potential_events_list = list(events_captured)
    freq_of_events = dict.fromkeys(potential_events_list, 0)
    freq_of_events['no_event'] = 0
    indices_of_events = dict.fromkeys(potential_events_list, [])
    indices_of_events['no_event'] = []
    for i, datapoint in enumerate(dataset):
        (_, label) = datapoint
        if not label:
            freq_of_events['no_event'] = freq_of_events['no_event'] + 1
            indices_of_events["no_event"] = indices_of_events["no_event"].append(i)
        else:
            any_relevant_event_detected = False
            for event in label:
                if event in events_captured:
                    freq_of_events[event] = freq_of_events[event] + 1
                    indices_of_events[event] = indices_of_events[event].append(i)
                    any_relevant_event_detected = True
                    break
            if not any_relevant_event_detected:
                freq_of_events['no_event'] = freq_of_events['no_event'] + 1
                indices_of_events["no_event"] = indices_of_events["no_event"].append(i)
                    
    return freq_of_events, indices_of_events

def get_dataset_for_model_train_and_eval(data_dir_path, events_captured):
    # Get the inputs
    with open(data_dir_path + "traces_data.pkl", "rb") as f:
        traces_data = pickle.load(f)
    state_list = list(map(lambda td: td['vectors'], traces_data))
    state_list_conc = np.concatenate(state_list)
    
    # Get the target labels
    label_list = []
    num_eps = len(traces_data)
    for ep in range(num_eps):
        sub_dir_path = data_dir_path + "trace_" + str(ep) + "/"
        with open(sub_dir_path + "events_2.pkl", "rb") as f:
            events = pickle.load(f)
        label_list.append(events)
    label_list_conc = np.concatenate(label_list)
    dataset = list(zip(state_list_conc, label_list_conc))
    random.shuffle(dataset)

    final_dataset = []

    # Perform downsampling on the majority class (where no event is observed)
    downsampled_dataset = []
    _, indices_of_events = _get_distribution_of_labels(dataset, events_captured)

    current_dataset_size = len(dataset)
    num_desired_samples = 500 # TODO: Change this hardcoded value to something more meaningful
    # no_event_points = [dataset[i] for i in indices_of_events['no_event']]
    no_event_points = []
    event_points = []

    for i in len(dataset):
        if i in indices_of_events['no_event']:
            no_event_points.append(dataset[i])
        else:
            event_points.append(dataset[i])

    downsampled_dataset.extend(event_points)
    downsampled_samples = resample(no_event_points, replace=False, n_samples=num_desired_samples, random_state=None)
    downsampled_dataset.extend(downsampled_samples)

    (states, labels) = zip(*downsampled_dataset)

    # Use SMOTE to upsample the data
    smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=5, n_jobs=None)  
    # Should I downsample before upsampling with SMOTE? More likelihood of noisy datapoints generated because we have so few non-empty labelled datapoints to begin with and now we are trying to match an incredibly high number (950 ish from 15)
    new_states, new_labels = smote.fit_resample(states, labels)
    upsampled_samples = zip(new_states, new_labels)
    final_dataset.extend(upsampled_samples)

    # Where do I specify by how much to upsample each class by?. It comes from the sampling_strategy, 'auto' meaning that every other class apart from the majority looks to be equalised

    random.shuffle(final_dataset)
    
    # Split up dataset into training and test datasets once shuffled
    final_data_size = len(final_dataset)
    cut_off = math.floor((2 * final_data_size) / 3)
    train_data = final_dataset[:cut_off]
    test_data = final_dataset[cut_off:]
    
    return train_data, test_data

def get_output_size(dynamic_balls):
    # 21 is obtained from n * (n - 1) / 2 where n is 7 (7 balls)
    return 6 if not dynamic_balls else 21

def run_labelling_func_framework():
    # Determines whether or not the balls are frozen
    use_velocities = False

    # Generate training data
    train_data_dir = "/vol/bitbucket/ras19/fyp/dataset2/training/"
    img_base_fname = "step"
    test_data_dir = "/vol/bitbucket/ras19/fyp/dataset2/test/"
    test_img_base_fname = "test_step"

    # generate_unlabelled_images(use_velocities, train_data_dir, img_base_fname)
    # train_set_dir_path, events_captured = label_dataset(train_data_dir, img_base_fname)
    # with open("events_captured_3.pkl", "wb") as f:
    #     pickle.dump(events_captured, f)

    # Generate test data
    # generate_unlabelled_images(use_velocities, test_data_dir, test_img_base_fname)
    # label_dataset(test_data_dir, test_img_base_fname)

    # TODO: Need to check quality of training and test dataset created by specified metrics

    # Should I be filtering the irrelevant events here?
    with open("events_captured_3.pkl", "rb") as f:
        events_captured = pickle.load(f)
    events_captured_filtered = set(filter(lambda pair: pair[0] == "black" or pair[1] == "black", events_captured))
    
    # Create the model (i.e. learnt labelling function)
    input_size = 52 if use_velocities else 28
    output_size = 21 if use_velocities else 6
    num_layers = 6
    num_neurons = 64
    labelling_fn = State2EventNet(input_size, output_size, num_layers, num_neurons)
    
    # Get the training and test data from what has (already) been generated
    train_data, test_data = get_dataset_for_model_train_and_eval(train_data_dir, events_captured_filtered)

    # print("EVALUATING THE TRAINING DATASET")
    # analyse_dataset(train_data, events_captured)
    # print("EVALUATING THE TEST DATASET")
    # analyse_dataset(test_data, events_captured)
    
    learning_rate = 0.01
    num_train_epochs = 500
    train_batch_size = 32
    test_batch_size = train_batch_size

    # Initialise weights and biases here
    wandb.init(
        project="labelling-function-learning",
        config={
            "learning_rate": learning_rate,
            "epochs": num_train_epochs,
            "num_layers": num_layers,
            "num_neurons": num_neurons 
        }
    )

    labelling_fn = train_model(labelling_fn, train_data, train_batch_size, test_data, test_batch_size, learning_rate, num_train_epochs, output_size, events_captured)

    labelling_fn_loc = "trained_model/label_fun.pth"

    # print("Evaluating the initial model (without any training)")
    # eval_model(labelling_fn, test_data, test_batch_size, events_captured, output_size)

    # print("Evaluating the model after being trained on the training dataset")
    # labelling_fn.load_state_dict(torch.load(labelling_fn_loc))
    # eval_model(labelling_fn, test_data, test_batch_size, events_captured, output_size)
    # torch.save(labelling_fn.state_dict(), labelling_fn_loc)

    return labelling_fn

if __name__ == "__main__":
    labelling_function = run_labelling_func_framework()
