import random
import gym
import os
import pickle

ball_area = None

# Chooses one of five actions randomly
def choose_action(seed):
    NUM_ACTIONS = 5
    random.seed(seed)
    action = random.randint(0, NUM_ACTIONS - 1)
    return action
    
# Runs an agent following a random policy against a particular WaterWorld task (specified by the environment passed as the first argument)
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

            events_observed.append(observations)
            if done:
                trace_data_i["length"] = t + 1
                trace_data_i["vectors"] = states
                trace_data_i["ground_truth"] = events_observed
                trace_data_list.append(trace_data_i)

        events_per_episode.append(events_observed)

    with open(dir_path + "traces_data.pkl", "wb") as f:
        pickle.dump(trace_data_list, f)

# Saves the state traces made through human demonstrations
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
        trace_data_i["ground_truth"] = events
        trace_data_list.append(trace_data_i)

    with open(dir_path + "traces_data.pkl", "wb") as f:
        pickle.dump(trace_data_list, f)

if __name__ == "__main__":
    use_velocities = False
    env = gym.make(
        "gym_subgoal_automata:WaterWorldDummy-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0, "episode_limit": 300},
    )
    random_dir_path = "../random_dataset/"
    img_base_filename = "step"
    random_seed = None
    num_episodes = 5
    manual_dir_path = "../manual_dataset/"
    
    # Generate unlabelled dataset with dummy task and manual policy
    run_rand_policy_and_save_traces(env, num_episodes, random_dir_path, img_base_filename, random_seed)
    
    save_traces_from_manual_play(env, num_episodes, manual_dir_path, img_base_filename)
