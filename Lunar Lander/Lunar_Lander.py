# !apt install swig cmake
# !pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt
# !sudo apt-get update
# !sudo apt-get install -y python3-opengl
# !apt install ffmpeg
# !apt install xvfb
# !pip3 install pyvirtualdisplay

import gymnasium as gym
from pyvirtualdisplay import Display
from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Virtual display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()


def simple_manipulation():
    # First, we create our environment called LunarLander-v2
    env = gym.make("LunarLander-v2")

    # Then we reset this environment
    observation, info = env.reset()

    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space Shape", env.observation_space.shape)
    print("Sample observation", env.observation_space.sample())

    print("\n _____ACTION SPACE_____ \n")
    print("Action Space Shape", env.action_space.n)
    print("Action Space Sample", env.action_space.sample())

    for _ in range(20):
        # Take a random action
        action = env.action_space.sample()
        print("Action taken:", action)

        # Do this action in the environment and get
        # next_state, reward, terminated, truncated and info
        observation, reward, terminated, truncated, info = env.step(action)

        # If the game is terminated (in our case we land, crashed) or truncated (timeout)
        if terminated or truncated:
            # Reset the environment
            print("Environment is reset")
            observation, info = env.reset()

    env.close()


def stable_baseline_training():

    env = make_vec_env("LunarLander-v2", n_envs=16)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=1000000)

    model_name = "ppo-LunarLander-v2"
    model.save(model_name)
    evaluate_model(model)


def evaluate_model(model):
    eval_env = Monitor(gym.make("LunarLander-v2", render_mode="rgb_array"))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


def push_to_hub(model, model_name):

    env_id = "LunarLander-v2"
    model_architecture = "PPO"
    repo_id = "eloiblt/ppo-LunarLander-v2"
    commit_message = "Upload PPO LunarLander-v2 trained agent v2"

    eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

    # PLACE the package_to_hub function you've just filled here
    package_to_hub(
        model=model,  # Our trained model
        model_name=model_name,  # The name of our trained model
        model_architecture=model_architecture,  # The model architecture we used: in our case PPO
        env_id=env_id,  # Name of the environment
        eval_env=eval_env,  # Evaluation Environment
        repo_id=repo_id,  # id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name} for instance ThomasSimonini/ppo-LunarLander-v2
        commit_message=commit_message,
    )


def login_to_hub():

    notebook_login()
    # !git config --global credential.helper store


def load_from_hub():

    repo_id = "eloiblt/ppo-LunarLander-v2"  # The repo_id
    filename = "ppo-LunarLander-v2.zip"  # The model filename.zip
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    checkpoint = load_from_hub(repo_id, filename)
    model = PPO.load(checkpoint, custom_objects=custom_objects, print_system_info=True)
    eval_env = Monitor(gym.make("LunarLander-v2"))
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
