import gymnasium as gym
import gym_pusht
from gymnasium.wrappers import RecordVideo

modality = "keypoints"
render_mode = "human"
render_mode = "rgb_array"

if render_mode == "human":
    env = gym.make("gym_pusht/PushT-v0", obs_type=modality, render_mode="human")
else:
    env = gym.make("gym_pusht/PushT-v0", obs_type=modality, render_mode="rgb_array")
    env = RecordVideo(env, video_folder="./videos")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # print(f"Observation: {observation}")
    image = env.render()

    if terminated or truncated:
        observation, info = env.reset()

env.close()
