from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, KMeansSMOTE
import numpy as np
import pickle
import math
import random
import gym
import wandb

# Constructs a mapping of the distribution of labels
def _get_distribution_of_labels(dataset, events_captured):
    potential_events_list = sorted(list(events_captured))
    freq_of_events = {event: 0 for event in potential_events_list}
    freq_of_events['no_event'] = 0
    indices_of_events = {event: [] for event in potential_events_list}
    indices_of_events['no_event'] = []

    for i, datapoint in enumerate(dataset):
        (_, label) = datapoint
        if not label:
            freq_of_events['no_event'] = freq_of_events['no_event'] + 1
            indices_of_events['no_event'].append(i)
        else:
            any_relevant_event_detected = False
            for event in label:
                if event in events_captured:
                    freq_of_events[event] = freq_of_events[event] + 1
                    indices_of_events[event].append(i)
                    any_relevant_event_detected = True
            if not any_relevant_event_detected:
                freq_of_events['no_event'] = freq_of_events['no_event'] + 1
                indices_of_events['no_event'].append(i)
    
    return freq_of_events, indices_of_events

# Analyse the label distribution of the dataset
def analyse_dataset(dataset, events_captured):
    print("The size of the dataset is {}".format(len(dataset)))
    label_distribution, _ = _get_distribution_of_labels(dataset, events_captured)
    print("The label distribution")
    print(label_distribution)

# Perform downsampling on the class where no events are observed (this is what appears to be in the vast majority)
def downsample_dataset(dataset, indices_of_events, num_desired_samples):
    downsampled_dataset = []

    curr_dataset_size = len(dataset)
    no_event_points = []
    event_points = []

    for i in range(curr_dataset_size):
        if i in indices_of_events['no_event']:
            no_event_points.append(dataset[i])
        else:
            event_points.append(dataset[i])

    downsampled_dataset.extend(event_points)
    downsampled_samples = resample(no_event_points, replace=False, n_samples=num_desired_samples, random_state=None)
    downsampled_dataset.extend(downsampled_samples)

    return downsampled_dataset

# SMOTE upsampling
def upsample_with_smote(dataset, events_captured, use_velocities, k_neighbours=5):
    (states, labels) = zip(*dataset)

    final_dataset = dataset
    for event in events_captured:
        print(event)
        smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=k_neighbours, n_jobs=None)
        binary_labels = [1 if event in label else 0 for label in labels]
        print("Binary labels before resampling with SMOTE")
        new_states, new_labels = smote.fit_resample(states, binary_labels)
        additional_states = new_states[len(states):]
        additional_labels = new_labels[len(labels):]
        event_set_label = set()
        event_set_label.add(event)
        additional_event_set_labels = [event_set_label] * len(additional_labels)
        new_samples = zip(additional_states, additional_event_set_labels)
        final_dataset.extend(new_samples)

    return final_dataset
    
# Upsampling with KMeans SMOTE
def upsample_with_kmeans_smote(dataset, events_captured, use_velocities, k_neighbours=2, num_clusters=130, cluster_balance_threshold=1.0):
    (states, labels) = zip(*dataset)

    final_dataset = dataset
    for event in events_captured:
        print(event)
        # KMeans SMOTE object initialised with the default values for each parameter. Needs to be changed
        kmeans_smote = KMeansSMOTE(sampling_strategy='auto', 
                                   random_state=None, 
                                   k_neighbors=k_neighbours, 
                                   n_jobs=None, 
                                   kmeans_estimator=num_clusters, cluster_balance_threshold=cluster_balance_threshold, density_exponent='auto')
        binary_labels = [1 if event in label else 0 for label in labels]
        new_states, new_labels = kmeans_smote.fit_resample(states, binary_labels)
        additional_states = new_states[len(states):]
        additional_labels = new_labels[len(labels):]
        event_set_label = set()
        event_set_label.add(event)
        additional_event_set_labels = [event_set_label] * len(additional_labels)
        new_samples = zip(additional_states, additional_event_set_labels)
        check_generated_samples(additional_states, use_velocities, event)
        final_dataset.extend(new_samples)

    return final_dataset

# Upsample minority classes in a dataset
def upsample_randomly(dataset, events_captured):
    initial_freq_of_events, _ = _get_distribution_of_labels(dataset, events_captured)
    num_desired_samples = max(list(initial_freq_of_events.values()))
    print(num_desired_samples)
    new_dataset = dataset.copy()
    for state, event_set in dataset:
        if event_set:
            for event in event_set:
                already_upsampled = False
                if event in events_captured and not already_upsampled:
                    curr_freq = initial_freq_of_events[event]
                    amount_to_upsample = math.floor((num_desired_samples - curr_freq)  / curr_freq)
                    for _ in range(amount_to_upsample):
                        new_dataset.append((state, event_set))
                    already_upsampled = True
    return new_dataset

# Check the labelling of synthetically generated states 
def check_generated_samples(new_samples, use_velocities, event):
    env = gym.make("gym_subgoal_automata:WaterWorldDummy-v0",
                   params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 200})
    img_dir_path = "synthetic_samples/"
    for i in range(2):
        print(new_samples[i])
        print("The event one should observe in this state is " + str(event))
        env.see_synthetic_state(new_samples[i], use_velocities, img_dir_path, str(event) + str(i))

# Obtains the dataset from the given directory 
def get_dataset(data_dir_path, events_captured, events_fname, use_velocities, see_dataset=True, is_test=True):
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
        with open(sub_dir_path + events_fname, "rb") as f:
            events = pickle.load(f)
        label_list.append(events)
    label_list_conc = np.concatenate(label_list)

    # Create initial dataset and get its metadata
    dataset = list(zip(state_list_conc, label_list_conc))
    random.shuffle(dataset)
    
    initial_freq_of_events, indices_of_events = _get_distribution_of_labels(dataset, events_captured)
    for event in initial_freq_of_events.keys():
        wandb.log({"event": event, "initial_freq": initial_freq_of_events[event]})

    if see_dataset:
        print("Before any data augmentation (downsampling or upsampling)")
        print(initial_freq_of_events)

    if not is_test:
        # Perform downsampling on the dataset
        dataset = downsample_dataset(dataset, indices_of_events, num_desired_samples=500)

        # Upsample the dataset in one of three ways
        # train_data = upsample_with_smote(train_data, events_captured, use_velocities, k_neighbours=5)
        # dataset = upsample_with_kmeans_smote(dataset, events_captured, use_velocities, k_neighbours=2, num_clusters=10, cluster_balance_threshold=2.0)
        dataset = upsample_randomly(dataset, events_captured)

    if see_dataset:
        print("After some data augmentation (downsampling or upsampling or both)")
        new_freq_of_events, _ = _get_distribution_of_labels(dataset, events_captured)
        print(new_freq_of_events)
        print("The size of the dataset is {}".format(len(dataset)))

    random.shuffle(dataset)

    freq_of_events, _ = _get_distribution_of_labels(dataset, events_captured)
    
    return dataset, freq_of_events