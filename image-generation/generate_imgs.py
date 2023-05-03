# Note that the playing of the environment to generate the images of the trace requires the use of the WaterWorld environment, which in turn requires Python version 3.7. This conflicts with the need for Python version 3.8 and above for Segment Anything. The generation of images and image segmentation of said images, in light of this, were treated as two separate tasks.
import gym

if __name__ == "__main__":
    use_velocities = False 
    # Initially have coloured balls frozen
    env = gym.make(
        "gym_subgoal_automata:WaterWorldRedGreen-v0",
        params={"generation": "random", "use_velocities": use_velocities, "environment_seed": 0},
    )
    img_dir_path = "../image-segmentation/ww_trace/"
    env.play(img_dir_path)