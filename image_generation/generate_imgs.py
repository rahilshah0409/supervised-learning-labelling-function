# Note that the playing of the environment to generate the images of the trace requires the use of the WaterWorld environment, which in turn requires Python version 3.7. This conflicts with the need for Python version 3.8 and above for Segment Anything. The generation of images and image segmentation of said images, in light of this, were treated as two separate tasks.
import random
import gym
import os

ball_area = None

def choose_action(seed):
    NUM_ACTIONS = 5
    random.seed(seed)
    action = random.randint(0, NUM_ACTIONS - 1)
    return action

def run_agent(env, num_episodes, random_seed=None):
    ep_durs = []
    states_traversed = []
    events_per_episode = []
    img_dir_path = "../image_segmentation/ww_trace_rand/"
    base_filename = "env_rand_step"
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))

        state = env.reset()
        # Some rendering and saving of the image you get. INSERT HERE
        env.save_and_render(img_dir_path, base_filename, 0)
        states = []
        states.append(state)

        events_observed = [set()]
        done, terminated, t = False, False, 1

        while not (done or terminated):
            action = choose_action(random_seed)
            if random_seed is not None:
                random_seed += 2
            next_state, _, done, observations = env.step(action)
            t += 1
            env.save_and_render(img_dir_path, base_filename, t)
            state = next_state
            states.append(state)

            # if (observations != set()):
            events_observed.append(observations)
            if done:
                ep_durs.append((t, ep))

        states_traversed.append(states)
        events_per_episode.append(events_observed)
    return ep_durs, states_traversed, events_per_episode

if __name__ == "__main__":
    use_velocities = False 
    # Initially have coloured balls frozen
    env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0},
    )
    # img_dir_path = "../image_segmentation/ww_trace/"
    # env.play(img_dir_path)
    run_agent(env, 1)
    # ball_area = env.get_ball_area()