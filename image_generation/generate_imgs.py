# Note that the playing of the environment to generate the images of the trace requires the use of the WaterWorld environment, which in turn requires Python version 3.7. This conflicts with the need for Python version 3.8 and above for Segment Anything. The generation of images and image segmentation of said images, in light of this, were treated as two separate tasks.
import random
import gym
import os
import pickle

ball_area = None

def choose_action(seed):
    NUM_ACTIONS = 5
    random.seed(seed)
    action = random.randint(0, NUM_ACTIONS - 1)
    return action
    
# Note that if you wanted to run multiple traces, only the last trace will be saved because you will be overriding each image in the same directory (same name). Ideally want this functionality to move up to the function above. Move onto this later because we are only running one trace (but maybe have sub directory for each trace and corresponding images?)
def run_rand_policy_and_save_traces(env, num_episodes, dir_path, img_base_filename, random_seed=None):
    events_per_episode = []
    trace_data_list = []
    for ep in range(num_episodes):
        sub_dir_path = dir_path + "trace_" + str(ep)
        if not os.path.exists(sub_dir_path):    
            os.mkdir(sub_dir_path)
        sub_dir_path = sub_dir_path + "/"
        trace_data_i = {
            "vectors": [],
            "length": 0
        }

        print("Episode {} in progress".format(ep + 1))

        state = env.reset()
        # Some rendering and saving of the image you get. INSERT HERE
        env.save_and_render(sub_dir_path, img_base_filename, 0)
        states = []
        states.append(state)

        events_observed = [set()]
        done, terminated, t = False, False, 0
        while not (done or terminated):
            action = choose_action(random_seed)
            if random_seed is not None:
                random_seed += 2
            next_state, _, done, observations = env.step(action, t)
            t += 1
            env.save_and_render(sub_dir_path, img_base_filename, t)
            state = next_state
            states.append(state)

            # if (observations != set()):
            events_observed.append(observations)
            if done:
                trace_data_i["length"] = t + 1
                trace_data_i["vectors"] = states
                trace_data_i["ground_truth"] = events_observed
                trace_data_list.append(trace_data_i)

        events_per_episode.append(events_observed)

    with open(dir_path + "traces_data.pkl", "wb") as f:
        pickle.dump(trace_data_list, f)

def save_traces_from_manual_play(env, num_episodes, dir_path, img_base_filename):
    trace_data_list = []
    for ep in range(num_episodes):
        sub_dir_path = dir_path + "trace_" + str(ep)
        if not os.path.exists(sub_dir_path):    
            os.mkdir(sub_dir_path)
        sub_dir_path = sub_dir_path + "/"
        trace_data_i = {
            "vectors": [],
            "length": 0
        }
        _, states, events, _ = env.play(sub_dir_path, img_base_filename)
        trace_data_i["length"] = len(states)
        trace_data_i["vectors"] = states
        trace_data_i["ground_truth_labels"] = events
        trace_data_list.append(trace_data_i)

    with open(dir_path + "traces_data.pkl", "wb") as f:
        pickle.dump(trace_data_list, f)

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

if __name__ == "__main__":
    # old_events_fname = "old_events.pkl"
    # final_events_fname = "final_events.pkl"
    # final_dataset_train_dir = "/vol/bitbucket/ras19/fyp/final_dataset/training/"
    # event_labelling_accuracy_old = check_quality_of_dataset(final_dataset_train_dir, old_events_fname)
    # event_labelling_accuracy_new = check_quality_of_dataset(final_dataset_train_dir, final_events_fname)

    # print("Accuracy of labelling with the OLD collision detection algorithm")
    # for event in event_labelling_accuracy_old:
    #     (correct, incorrect) = event_labelling_accuracy_old[event]
    #     recall = correct / (correct + incorrect)
    #     print("Event: " + event + " Recall: {}".format(recall))

    # print("Accuracy of labelling with the NEW collision detection algorithm")
    # for event in event_labelling_accuracy_new:
    #     (correct, incorrect) = event_labelling_accuracy_new[event]
    #     recall = correct / (correct + incorrect)
    #     print("Event: " + event + " Recall: {}".format(recall))
    use_velocities = False
    # Initially have coloured balls frozen
    env = gym.make(
        "gym_subgoal_automata:WaterWorldDummy-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 250},
    )
    random_dir_path = "/vol/bitbucket/ras19/fyp/random_dataset/"
    img_base_filename = "step"
    random_seed = None
    num_episodes = 5
    run_rand_policy_and_save_traces(env, num_episodes, random_dir_path, img_base_filename, random_seed)
    # with open(dir_path + "traces_data.pkl", "rb") as f:
    #     trace_data = pickle.load(f)
    # print(trace_data[0]['length'])

    # manual_dir_path = "/vol/bitbucket/ras19/fyp/manual_dataset/"
    # save_traces_from_manual_play(env, num_episodes, manual_dir_path, img_base_filename)
    # img_dir_path = "../image_segmentation/ww_trace/"
    # env.play(img_dir_path)
    # ball_area = env.get_ball_area()
    # env.play()
