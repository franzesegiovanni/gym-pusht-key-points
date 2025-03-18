import numpy as np
import gymnasium as gym
import pickle
import os
import argparse

#!/usr/bin/env python3

def play_demonstration(demo_path):
    # Load the demonstration file
    with open(demo_path, 'rb') as f:
        demo_data = pickle.load(f)
    
    # Extract observations, actions, etc. from the demonstration
    # The exact structure depends on how your demonstrations are saved
    observations = demo_data.get('observations', [])
    actions = demo_data.get('actions', [])
    
    if not observations or not actions:
        print("Error: Demonstration file doesn't contain observations or actions")
        return
    
    # Create the environment - adjust env_name to match your actual environment
    env = gym.make('PushT-v0')
    
    # Reset the environment
    obs = env.reset()
    
    # Set the initial state to match the first state in the demonstration
    # This depends on the environment's API for setting states
    try:
        env.set_state(observations[0])
        print("Environment initialized to demonstration's first state")
    except AttributeError:
        print("Warning: Environment doesn't support setting state directly")
    
    # Play through the demonstration
    total_reward = 0
    for i, action in enumerate(actions):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()  # Visualize each step
        
        print(f"Step {i+1}/{len(actions)}, Action: {action}, Reward: {reward}")
        
        if done:
            print("Environment signaled done")
            break
    
    print(f"Demonstration playback complete. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--demo_path", type=str, required=True, 
                        help="Path to the demonstration pickle file")
    args = parser.parse_args()
    
    play_demonstration(args.demo_path)