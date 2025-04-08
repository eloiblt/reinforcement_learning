import gymnasium as gym
import numpy as np
import json

maze_size = "8x8"  # 4x4, 8x8

def train(runs):

    env = gym.make(
        "FrozenLake-v1",
        render_mode=None,
        map_name=maze_size,
        is_slippery=False,  # for deterministic or non deterministic search
    )

    # observation space = number of states : 0-15 (4x4), 0-63 (8x8)
    # action space : 0 left, 1 down, 2 right, 3 up
    # q-table : 16 states x 4 actions (4x4), 64 states x 4 actions (8x8)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    learning_rate = 0.1  # 0.9 = 90% learning rate, 0.1 = 10% learning rate. Near 0 = slow learning, near 1 = fast learning
    discount_factor = 0.99  # 0.9 = 90% discount factor, 0.1 = 10% discount factor. Near 0 = short term, near 1 = long term
    epsilon = 1  # 1 = 100% exploration, 0 = 100% exploitation
    epsilon_decay = (
        0.00005  # decay rate of exploration (1 to 0) 1 / 0.0001 = 10000 episodes
    )
    rng = np.random.default_rng()

    for _ in range(runs):
        state = env.reset()[0]
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = (
                    env.action_space.sample()
                )  # exploration, random action 0 left, 1 down, 2 right, 3 up
            else:
                action = np.argmax(q_table[state])  # exploitation

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Bellman equation
            # Q(s,a) = Q(s,a) + alpha * (r + gamma * max_a Q(s',a') - Q(s,a))
            # Where : Q(s,a) = Q-value of state s and action a
            # alpha = learning rate
            # r = reward received
            # gamma = discount factor
            # max_a Q(s',a') = maximum Q-value of state s' for all actions a'
            q_table[state, action] = q_table[state, action] + learning_rate * (
                reward
                + discount_factor * np.max(q_table[new_state, :])
                - q_table[state, action]
            )
            state = new_state

        epsilon = max(0.01, epsilon - epsilon_decay)

        if epsilon == 0:
            learning_rate = 0.0001

    env.close()

    with open("q_table.json", "w") as f:
        json.dump(q_table.tolist(), f)

    print("Training finished")


def run():

    env = gym.make(
        "FrozenLake-v1",
        render_mode="human",
        map_name=maze_size,
        is_slippery=False,
    )

    with open("q_table.json", "r") as f:
        q_table = np.array(json.load(f))

    state = env.reset()[0]
    terminated = False

    while not terminated:
        action = np.argmax(q_table[state])  # exploitation

        new_state, reward, terminated, truncated, _ = env.step(action)
        state = new_state

    env.close()


if __name__ == "__main__":
    print("Training...")
    train(15000)
    print("Running...")
    run()
