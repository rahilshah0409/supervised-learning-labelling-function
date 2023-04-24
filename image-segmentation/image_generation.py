# A random policy
import random
import gym

def choose_action(seed):
    NUM_ACTIONS = 5
    random.seed(seed)
    action = random.randint(0, NUM_ACTIONS - 1)
    return action


def run_agent(env, num_episodes, random_seed=None):
    ep_durs = []
    states_traversed = []
    events_per_episode = []
    for ep in range(num_episodes):
        print("Episode {} in progress".format(ep + 1))

        state = env.reset()
        # Some rendering and saving of the image you get. INSERT HERE
        states = []
        states.append(state)

        events_observed = [set()]
        done, terminated, t = False, False, 1

        while not (done or terminated):
            action = choose_action(random_seed)
            if random_seed is not None:
                random_seed += 2
            next_state, _, done, observations = env.step(action)
            state = next_state
            states.append(state)
            t += 1

            # if (observations != set()):
            events_observed.append(observations)
            if done:
                ep_durs.append((t, ep))

        states_traversed.append(states)
        events_per_episode.append(events_observed)
    return ep_durs, states_traversed, events_per_episode

def play_with_env(env):
    env.reset()
    env.render()

if __name__ == "__main__":
    print("This file is supposed to be generated images from a random policy")
    fixed_balls_env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "random_restart": False, "environment_seed": 0},
    )
    play_with_env(fixed_balls_env)