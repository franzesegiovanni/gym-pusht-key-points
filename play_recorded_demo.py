import numpy as np
import gymnasium as gym
import pickle
import os
import argparse
import gym_pusht
import pygame
from utils import read_point_distribution
#!/usr/bin/env python3

def play_demonstration(demo_path, env_name="gym_pusht/PushT-v0", obs_type="keypoints", render_mode="human"):
    # Load the demonstration file
    with open(demo_path, 'rb') as f:
        demo_data = pickle.load(f)

    env = gym.make(env_name, obs_type=obs_type, render_mode=render_mode)
    
    # reset to a specific state
    object_state=demo_data[0]['observation']['object_state']
    agent_state=demo_data[0]['observation']['agent_pos']
    goal_state=demo_data[0]['observation']['goal_state']
    keypoint_object=demo_data[0]['observation']['object_keypoints']
    keypoint_goal = demo_data[0]['observation']['goal_keypoints']
    keypoint_object = np.array(keypoint_object).reshape(-1, 2)
    keypoint_goal = np.array(keypoint_goal).reshape(-1, 2)
    distribution = np.vstack((keypoint_object, keypoint_goal,agent_state))
    print("Distribution shape:", distribution.shape)
    obs, info = env.reset(options={"reset_to_state": [agent_state[0], agent_state[1], object_state[0], object_state[1], object_state[2], goal_state[0], goal_state[1], goal_state[2]]})
    #choose an arbitrary starting state
    print("reset to state", [agent_state[0], agent_state[1], object_state[0], object_state[1], object_state[2], goal_state[0], goal_state[1], goal_state[2]])
    total_reward = 0
    
    for i in range(1,len(demo_data)):
        action = demo_data[i]['action']
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        distribution=read_point_distribution(next_obs) 
        # print(distribution.shape)
        env.render()  # Visualize each step
        pygame.time.Clock().tick(20)  # Limit to 60 FPS
    
    print(f"Demonstration playback complete. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--demo_path", type=str, required=True, 
                        help="Path to the demonstration pickle file")
    args = parser.parse_args()
    
    play_demonstration(args.demo_path)