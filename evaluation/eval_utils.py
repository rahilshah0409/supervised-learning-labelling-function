import pickle

# Creates a dictionary that records the number of correct and incorrect predictions for each event (including no event)
def check_quality_of_dataset(data_dir, events_fname):
    trace_data_filename = "traces_data.pkl"
    with open(data_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    accuracy_of_labelling = {'r': (0, 0), 'c': (0, 0), 'g': (0, 0), 'y': (0, 0), 'm': (0, 0), 'b': (0, 0), 'none': (0, 0)}

    for ep in range(len(trace_data)):
        sub_dir = data_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        with open(sub_dir + events_fname, "rb") as f:
            labelled_events = pickle.load(f)
        ground_truth_events = trace_data[ep]["ground_truth_labels"]
        for step in range(ep_len):
            if _events_equivalent(labelled_events[step], ground_truth_events[step]):
                if not ground_truth_events[step]:
                    (correct, incorrect) = accuracy_of_labelling['none']
                    accuracy_of_labelling = (correct + 1, incorrect)
                else:
                    for event in ground_truth_events[step]:
                        (correct, incorrect) = accuracy_of_labelling[event]
                        accuracy_of_labelling = (correct + 1, incorrect)
            else:
                if not ground_truth_events[step]:
                    (correct, incorrect) = accuracy_of_labelling['none']
                    accuracy_of_labelling = (correct, incorrect + 1)
                else:
                    for event in ground_truth_events[step]:
                        (correct, incorrect) = accuracy_of_labelling[event]
                        accuracy_of_labelling = (correct, incorrect + 1)
    
    return accuracy_of_labelling

# Helper method to check if a ground truth event observed has been labelled
def _events_equivalent(detected_events, ground_truth):
    if not ground_truth and not detected_events:
        return True
    elif len(detected_events) != len(ground_truth):
        return False
    else:
        for (agent, coloured) in detected_events:
            if coloured[0] not in ground_truth and coloured[0] == 'l' and 'g' not in ground_truth:
                return False
        return True

# Creates a frequency distribution of the events observed in the traces that make up a dataset. This looks at the ground truth labels and not our labelling of the state.
def get_original_dataset_distribution(data_dir_path):
    with open(data_dir_path + "traces_data.pkl", "rb") as f:
        traces_data = pickle.load(f)
    distribution = {'r': 0, 'g': 0, 'c': 0, 'm': 0, 'y': 0, 'b': 0, 'none': 0}
    num_eps = len(traces_data)
    for ep in range(num_eps):
        print(traces_data[ep])
        events_ep = traces_data[ep]["ground_truth"]
        for event_set in events_ep:
            if not event_set:
                distribution['none'] += 1
            else:
                for event in event_set:
                    distribution[event] += 1
    events = sorted(set(distribution.keys()))
    for event in events:
        freq = distribution[event]
        print("Event: " + str(event) + " Frequency: {}".format(freq))

# Checks how accurately a dataset has been labelled by comparing each label to the ground truth label
def accuracy_of_labelling(events_fname, data_dir):
    event_labelling_accuracy = check_quality_of_dataset(data_dir, events_fname)

    for event in event_labelling_accuracy:
        (correct, incorrect) = event_labelling_accuracy[event]
        recall = correct / (correct + incorrect)
        print("Event: " + event + " Recall: {}".format(recall))