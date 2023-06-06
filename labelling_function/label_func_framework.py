import sys

from dataset_augmentation import get_dataset
sys.path.insert(1, "../")
from image_segmentation.dataset_labelling_pipeline import generate_event_labels_from_masks, generate_and_save_masks_for_eps
from image_generation.generate_imgs import run_rand_policy_and_save_traces
from labelling_model import State2EventNet
from model_training import eval_model, train_model
import wandb
import gym
import pickle
import torch

def generate_unlabelled_images(use_velocities, dataset_dir_path, img_base_fname):
    env = gym.make("gym_subgoal_automata:WaterWorldDummy-v0",
                   params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 200})
    random_seed = None
    num_episodes = 5

    # Generate training data without labels (images and metadata)
    run_rand_policy_and_save_traces(env, num_episodes, dataset_dir_path, img_base_fname, random_seed)

def label_dataset(img_dir_path, img_base_fname, events_fname):
    # Segment the images with Segment Anything
    trace_data_filename = "traces_data.pkl"
    with open(img_dir_path + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    
    # The following lines need to be run on a lab machine
    # sam_checkpoint = "/vol/bitbucket/ras19/fyp/se-model-checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    filtered_masks_pkl_name_base = "masks"
    # masks_for_every_ep = generate_and_save_masks_for_eps(trace_data, img_dir_path, sam_checkpoint, model_type, img_base_fname, filtered_masks_pkl_name_base)

    # Run algorithm to get the events for each state generated. num_events should be changed here based on empirical analysis on the training data and the labelling done on it
    events_for_every_ep, events_observed = generate_event_labels_from_masks(trace_data, img_dir_path, model_type, filtered_masks_pkl_name_base, img_base_fname, events_fname)
    return img_dir_path, events_observed

def get_output_size(dynamic_balls):
    # 21 is obtained from n * (n - 1) / 2 where n is 7 (7 balls)
    return 6 if not dynamic_balls else 21

def check_quality_of_dataset(data_dir, events_fname):
    trace_data_filename = "traces_data.pkl"
    with open(data_dir + trace_data_filename, "rb") as f:
        trace_data = pickle.load(f)
    no_of_correct_labels = 0
    no_of_incorrect_labels = 0

    for ep in range(len(trace_data)):
        sub_dir = data_dir + "trace_" + str(ep) + "/"
        ep_len = trace_data[ep]["length"]
        with open(sub_dir + events_fname, "rb") as f:
            labelled_events = pickle.load(f)
        ground_truth_events = trace_data[ep]["ground_truth_labels"]
        for step in range(ep_len):
            if _events_equivalent(labelled_events[step], ground_truth_events[step]):
                no_of_correct_labels += 1
            else:
                no_of_incorrect_labels += 1
                print(ground_truth_events[step])
                print(labelled_events[step])
    
    return no_of_correct_labels, no_of_incorrect_labels

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
    

def run_labelling_func_framework():
    # Determines whether or not the balls are frozen
    use_velocities = False

    # Generate training data
    train_data_dir = "/vol/bitbucket/ras19/fyp/fixed_seed_manual_dataset/training/"
    img_base_fname = "step"
    train_events_fname = "new_events.pkl"
    test_data_dir = "/vol/bitbucket/ras19/fyp/fixed_seed_manual_dataset/test/"
    test_img_base_fname = "test_step"
    test_events_fname = "new_test_events.pkl"

    # generate_unlabelled_images(use_velocities, train_data_dir, img_base_fname)
    train_set_dir_path, events_captured = label_dataset(train_data_dir, img_base_fname, train_events_fname)
    with open("events_captured_fixed_seed_final.pkl", "wb") as f:
        pickle.dump(events_captured, f)

    # Generate test data
    # generate_unlabelled_images(use_velocities, test_data_dir, test_img_base_fname, test_events_fname)
    label_dataset(test_data_dir, test_img_base_fname, test_events_fname)

    # Should I be filtering the irrelevant events here?
    with open("events_captured_fixed_seed_final.pkl", "rb") as f:
        events_captured = pickle.load(f)
    events_captured_filtered = sorted(list(filter(lambda pair: (pair[0] == "black" and pair[1] != "black") or (pair[0] != "black" and pair[1] == "black"), events_captured)))
    # print(events_captured_filtered)

    # TODO: Need to check quality of training and test dataset created by specified metrics
    #no_of_correct_labels, no_of_incorrect_labels = check_quality_of_dataset(train_data_dir, train_events_fname)
    #print("Correct labels: {}. Incorrect labels: {}".format(no_of_correct_labels, no_of_incorrect_labels))

    # Create the model (i.e. learnt labelling function)
    #input_size = 52 if use_velocities else 28
    #output_size = 21 if use_velocities else 6
    #num_layers = 6
    #num_neurons = 64
    #labelling_fn = State2EventNet(input_size, output_size, num_layers, num_neurons)

    #learning_rate = 0.01
    #num_train_epochs = 500
    #train_batch_size = 32
    #test_batch_size = train_batch_size

    # Initialise weights and biases here
    #wandb.init(
    #    project="effect_of_fixed_seed",
    #    config={
    #        "learning_rate": learning_rate,
    #        "epochs": num_train_epochs,
    #        "num_layers": num_layers,
    #        "num_neurons": num_neurons 
    #    }
    #)
    
    # Get the training and test data from what has (already) been generated
    #train_data, train_label_distribution = get_dataset(train_data_dir, events_captured_filtered, train_events_fname, use_velocities, see_dataset=False, is_test=False)
    #test_data, test_label_distribution = get_dataset(test_data_dir, events_captured_filtered, test_events_fname, use_velocities, see_dataset=False, is_test=True)

    #for event in train_label_distribution.keys():
    #    wandb.log({"event": event, "train_freq": train_label_distribution[event]})

    #for event in test_label_distribution.keys():
    #    wandb.log({"event": event, "test_freq": test_label_distribution[event]})
        
    #labelling_fn, precision_scores, recall_scores, f1_scores = train_model(labelling_fn, train_data, train_batch_size, test_data, test_batch_size, learning_rate, num_train_epochs, output_size, events_captured_filtered)

    model_dir = "trained_model/"
    base_model_name = "model_fixed_seed"
    labelling_fn_loc = model_dir + base_model_name + ".pth"
    #torch.save(labelling_fn.state_dict(), labelling_fn_loc)
    
    #model_metrics = {"precision": precision_scores,
    #                 "recall": recall_scores,
    #                 "f1": f1_scores}
    #metrics_loc = model_dir + base_model_name + "_metrics.pkl"
    #with open(metrics_loc, "wb") as f:
    #    pickle.dump(model_metrics, f)

    # print("Evaluating the initial model (without any training)")
    # eval_model(labelling_fn, test_data, test_batch_size, events_captured, output_size)

    # print("Evaluating the model after being trained on the training dataset")
    # labelling_fn.load_state_dict(torch.load(labelling_fn_loc))
    # avg_loss, precision_scores, recall_scores, f1_scores = eval_model(labelling_fn, test_data, test_batch_size, events_captured_filtered, output_size)

    # wandb.log({"loss": avg_loss})
    # print("Average loss: {}".format(avg_loss))
    # print("Distribution of labels")
    # print(str(test_label_distribution))
    # for event in precision_scores.keys():
    #     wandb.log({"event": str(event), "precision": precision_scores[event], "recall": recall_scores[event], "f1": f1_scores[event]})
    #     print("Event: " + str(event))
    #     print("Precision: {}".format(precision_scores[event]))
    #     print("Recall: {}".format(recall_scores[event]))
    #     print("F1: {}".format(f1_scores[event]))

    #return labelling_fn

if __name__ == "__main__":
    labelling_function = run_labelling_func_framework()
    # env = gym.make("gym_subgoal_automata:WaterWorldDummy-v0",
    #                params={"generation": "random", "use_velocities": True, "environment_seed": 0, "episode_limit": 200})
    # env.play()