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

if __name__ == "__main__":
    use_velocities = True
    # Initially have coloured balls frozen
    env = gym.make(
        "gym_subgoal_automata:WaterWorldDummy-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 200},
    )
    dir_path = "../image_segmentation/single_trace_experimentation/dynamic_balls/"
    img_base_filename = "step"
    random_seed = None
    num_episodes = 1
    # run_rand_policy_and_save_traces(env, num_episodes, dir_path, img_base_filename, random_seed)
    # with open(dir_path + "traces_data.pkl", "rb") as f:
    #     trace_data = pickle.load(f)
    # print(trace_data[0]['length'])
    save_traces_from_manual_play(env, num_episodes, dir_path, img_base_filename)
    # img_dir_path = "../image_segmentation/ww_trace/"
    # env.play(img_dir_path)
    # ball_area = env.get_ball_area()
    # env.play()