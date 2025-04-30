import numpy as np
import gymnasium as gym
import pickle
import os
import argparse
import gym_pusht
import pygame
#!/usr/bin/env python3

def play_demonstration(demo_path, env_name="gym_pusht/PushT-v0", obs_type="keypoints", render_mode="human"):
    # Load the demonstration file
    with open(demo_path, 'rb') as f:
        demo_data = pickle.load(f)
    demo_data= demo_data[0]
    # Extract observations, actions, etc. from the demonstration
    # The exact structure depends on how your demonstrations are saved
    # breakpoint()
    # Create the environment - adjust env_name to match your actual environment
    # env = gym.make('PushT-v0')


    env = gym.make(env_name, obs_type=obs_type, render_mode=render_mode)
    
    # Reset the environment
    obs = env.reset()
    # reset to a specific state
    # breakpoint()
    object_state=demo_data[0]['observation']['object_state']
    agent_state=demo_data[0]['observation']['agent_pos']
    goal_state=demo_data[0]['observation']['goal_state']
    obs, info = env.reset(options={"reset_to_state": [agent_state[0], agent_state[1], object_state[0], object_state[1], object_state[2], goal_state[0], goal_state[1], goal_state[2]]})
    #choose an arbitrary starting state
    print("reset to state", [agent_state[0], agent_state[1], object_state[0], object_state[1], object_state[2], goal_state[0], goal_state[1], goal_state[2]])
    # Set the initial state to match the first state in the demonstrationa    
    # Play through the demonstration
    total_reward = 0
    for i in range(1,len(demo_data)):
        action = demo_data[i]['action']
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        #print the goal state of the environment
        state=env.get_obs()
        env.render()  # Visualize each step

        if terminated:
            print("Environment signaled done")
            break
        pygame.time.Clock().tick(60)  # Limit to 60 FPS
    
    print(f"Demonstration playback complete. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--demo_path", type=str, required=True, 
                        help="Path to the demonstration pickle file")
    args = parser.parse_args()
    
    play_demonstration(args.demo_path)