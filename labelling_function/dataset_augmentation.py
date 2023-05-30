from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, KMeansSMOTE
import numpy as np
import pickle
import math
import random
import gym
import wandb

def _get_distribution_of_labels(dataset, events_captured):
    potential_events_list = sorted(list(events_captured))
    freq_of_events = {event: 0 for event in potential_events_list}
    freq_of_events['no_event'] = 0
    indices_of_events = {event: [] for event in potential_events_list}
    indices_of_events['no_event'] = []

    # print(len(dataset))
    # def process_datapoint(i, datapoint):
    #     nonlocal freq_of_events, indices_of_events
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

# This function should analyse the following:
# How much of each label appears
# The size
# Accuracy of labelling- how does accuracy of labelling translate when we have potentially irrelevant events- would we calculate precision, recall and F1 score?
# Anything else?
# The dataset passed in as an argument in this function has datapoints in the form of tuples 
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

# Questions
# If we are upsampling each class iteratively, do the new samples count to the dataset that we upsample in the next iteration. In the below implementation, this is not the case and this makes sense in my head because you don't want synthetically generated datapoints that might be noisy to determine how much the next class is upsampled by
# This implementation makes the most sense to me given this setting, but it is worth thinking about how viable and extensible this is to the dynamic setting, where multiple labels per state is more likely (I am not even sure if this downsampling upsampling business will be needed as much)
# Hyperparameters to think about: k_neighbours to sample b given reference a
def upsample_with_smote(dataset, events_captured, use_velocities, k_neighbours=5):
    (states, labels) = zip(*dataset)

    final_dataset = dataset
    for event in events_captured:
        print(event)
        smote = SMOTE(sampling_strategy='auto', random_state=None, k_neighbors=k_neighbours, n_jobs=None)
        binary_labels = [1 if event in label else 0 for label in labels]
        print("Binary labels before resampling with SMOTE")
        # print(binary_labels)
        new_states, new_labels = smote.fit_resample(states, binary_labels)
        additional_states = new_states[len(states):]
        additional_labels = new_labels[len(labels):]
        # print("Binary labels after resampling with SMOTE")
        # print(additional_labels)
        event_set_label = set()
        event_set_label.add(event)
        additional_event_set_labels = [event_set_label] * len(additional_labels)
        new_samples = zip(additional_states, additional_event_set_labels)
        # check_generated_samples(additional_states, use_velocities, event)
        final_dataset.extend(new_samples)

    return final_dataset
    
# Hyperparameters to think about: number of neighbours to sample b from reference a once a cluster has been selected for SMOTE upsampling, number of clusters to cluster the space on, the threshold that determines which clusters are selected for SMOTE upsampling
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
        # print("Binary labels before resampling with SMOTE")
        # print(binary_labels)
        new_states, new_labels = kmeans_smote.fit_resample(states, binary_labels)
        # print("Binary labels after resampling with SMOTE")
        # print(new_labels)
        additional_states = new_states[len(states):]
        additional_labels = new_labels[len(labels):]
        # print(additional_labels)
        event_set_label = set()
        event_set_label.add(event)
        additional_event_set_labels = [event_set_label] * len(additional_labels)
        new_samples = zip(additional_states, additional_event_set_labels)
        check_generated_samples(additional_states, use_velocities, event)
        final_dataset.extend(new_samples)

    return final_dataset

def upsample_randomly(dataset, events_captured):
    # print("The size of the dataset is: {}".format(len(dataset)))
    initial_freq_of_events, _ = _get_distribution_of_labels(dataset, events_captured)
    # print(initial_freq_of_events)
    num_desired_samples = max(list(initial_freq_of_events.values()))
    print(num_desired_samples)
    new_dataset = dataset.copy()
    for state, event_set in dataset:
        # print("Datapoint {}".format(i))
        # new_dataset.append((state, event_set))
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

def check_generated_samples(new_samples, use_velocities, event):
    env = gym.make("gym_subgoal_automata:WaterWorldDummy-v0",
                   params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 200})
    img_dir_path = "synthetic_samples/"
    for i in range(2):
        print(new_samples[i])
        # print("The event one should observe in this state is " + str(event))
        # env.see_synthetic_state(new_samples[i], use_velocities, img_dir_path, str(event) + str(i))

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
    # random.shuffle(dataset)
    initial_freq_of_events, indices_of_events = _get_distribution_of_labels(dataset, events_captured)

    if see_dataset:
        print("Before any data augmentation (downsampling or upsampling)")
        print(initial_freq_of_events)

    for event in initial_freq_of_events.keys():
        wandb.log({"event": event, "initial_freq": initial_freq_of_events[event]})

    if not is_test:
        # Perform downsampling on the dataset
        dataset = downsample_dataset(dataset, indices_of_events, num_desired_samples=250)

        # Upsample the dataset in one of three ways
        dataset = upsample_with_smote(dataset, events_captured, use_velocities, k_neighbours=5)
        # dataset = upsample_with_kmeans_smote(dataset, events_captured, use_velocities, k_neighbours=2, num_clusters=10, cluster_balance_threshold=2.0)
        # dataset = upsample_randomly(dataset, events_captured)

    if see_dataset:
        print("After some data augmentation (downsampling or upsampling or both)")
        new_freq_of_events, new_indices = _get_distribution_of_labels(dataset, events_captured)
        print(new_freq_of_events)
        print("The size of the dataset is {}".format(len(dataset)))

    random.shuffle(dataset)
    
    # Split up dataset into training and test datasets once shuffled
    # final_data_size = len(dataset)
    # cut_off = math.floor((2 * final_data_size) / 3)
    # train_data = dataset[:cut_off]
    # test_data = dataset[cut_off:]

    freq_of_events, _ = _get_distribution_of_labels(dataset, events_captured)
    
    return dataset, freq_of_events

# Thoughts on dataset handling, downsampling, upsampling, etc
# Should I downsample before upsampling with SMOTE? More likelihood of noisy datapoints generated because we have so few non-empty labelled datapoints to begin with and now we are trying to match an incredibly high number (950 ish from 15)
# Being careful when upsampling randomly because I know that we have a multi-label setting, meaning that copying one datapoint to upsample a certain class may also upsample another class indirectly. Upsampling in the multi-label setting needs a bit of thinking, but I am going to go with this because I think we don't have to worry too much about the multi-label setting (in this frozen environment)
# Having trouble applying kmeans smote to our particular dataset (my guess is because there is so much of an imbalance so no sufficient clusters can be formed to then be chosen for areas to oversampling on via SMOTE)
# In both SMOTE and KMeans SMOTE, I need to convert labels of datapoints to be binary and what is outputed is also binary. The new datapoints may have noise (and not be an interesection) but they may also be more than one intersection that I have no way (right now) of picking up and so I do a rudimentary conversion
# I am making the assumption in SMOTE and KMeans SMOTE that the 'new' datapoints are just added to the end of what already exists in the dataset. I don't know if this is true or not
# Where do I specify by how much to upsample each class by in SMOTE and KMeans SMOTE? It comes from the sampling_strategy, 'auto' meaning that every other class apart from the majority looks to be equalised
# Was having a problem with upsampling randomly and not getting the distribution of frequencies that I was expecting. The problem lies in the fact that in four of the initial states, multiple events are observed. When we go to upsample one of said labels (that needs to be upsampled by n), we are also upsampling the other label/s by n as well, even if that label needs to be upsampled by m instead. Not sure how to get around this at first glance but don't think it is a major issue for the purposes of balancing the dataset.