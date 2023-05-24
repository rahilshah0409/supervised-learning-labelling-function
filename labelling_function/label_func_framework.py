import sys

from dataset_augmentation import get_dataset_for_model_train_and_eval
sys.path.insert(1, "../")
from image_segmentation.dataset_labelling_pipeline import generate_event_labels_from_masks
from image_generation.generate_imgs import run_rand_policy_and_save_traces, save_traces_from_manual_play
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
    # run_rand_policy_and_save_traces(env, num_episodes, dataset_dir_path, img_base_fname, random_seed)
    save_traces_from_manual_play(env, num_episodes, dataset_dir_path, img_base_fname)

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

def get_output_size(dynamic_balls):
    # 21 is obtained from n * (n - 1) / 2 where n is 7 (7 balls)
    return 6 if not dynamic_balls else 21

def run_labelling_func_framework():
    # Determines whether or not the balls are frozen
    use_velocities = False

    # Generate training data
    train_data_dir = "/vol/bitbucket/ras19/fyp/dataset2/training/"
    img_base_fname = "step"
    test_data_dir = "../dataset/test/"
    test_img_base_fname = "test_step"

    # generate_unlabelled_images(use_velocities, train_data_dir, img_base_fname)
    # train_set_dir_path, events_captured = label_dataset(train_data_dir, img_base_fname)
    # with open("events_captured_3.pkl", "wb") as f:
    #     pickle.dump(events_captured, f)

    # Generate test data
    generate_unlabelled_images(use_velocities, test_data_dir, test_img_base_fname)
    # label_dataset(test_data_dir, test_img_base_fname)

    # TODO: Need to check quality of training and test dataset created by specified metrics

    # Should I be filtering the irrelevant events here?
    # with open("events_captured_3.pkl", "rb") as f:
    #     events_captured = pickle.load(f)
    # events_captured_filtered = sorted(list(filter(lambda pair: pair[0] == "black" or pair[1] == "black", events_captured)))
    
    # # Create the model (i.e. learnt labelling function)
    # input_size = 52 if use_velocities else 28
    # output_size = 21 if use_velocities else 6
    # num_layers = 6
    # num_neurons = 64
    # labelling_fn = State2EventNet(input_size, output_size, num_layers, num_neurons)

    # learning_rate = 0.01
    # num_train_epochs = 500
    # train_batch_size = 32
    # test_batch_size = train_batch_size
    
    # # Get the training and test data from what has (already) been generated
    # train_data, train_label_distribution, test_data, test_label_distribution = get_dataset_for_model_train_and_eval(train_data_dir, events_captured_filtered, use_velocities, see_dataset=False)

    # Initialise weights and biases here
    # wandb.init(
    #     project="effect_of_data_augmentation",
    #     config={
    #         "learning_rate": learning_rate,
    #         "epochs": num_train_epochs,
    #         "num_layers": num_layers,
    #         "num_neurons": num_neurons 
    #     }
    # )

    # for event in train_label_distribution.keys():
    #     wandb.log({"event": event, "freq": train_label_distribution[event]})

    # labelling_fn = train_model(labelling_fn, train_data, train_batch_size, test_data, test_batch_size, learning_rate, num_train_epochs, output_size, events_captured_filtered)

    # labelling_fn_loc = "trained_model/labelling_fn_down_250_upsample_smote.pth"

    # torch.save(labelling_fn.state_dict(), labelling_fn_loc)

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

    # return labelling_fn

if __name__ == "__main__":
    labelling_function = run_labelling_func_framework()