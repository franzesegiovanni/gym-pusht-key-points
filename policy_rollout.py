#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
import pickle
import os
import argparse
import gym_pusht
import pygame
from utils import read_point_distribution

from policy_transportation.transportation.transportation import PolicyTransportation
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations

delta_action_lim= 5.0
def get_keyboard_action():
    """Convert keyboard input to action"""
    action = np.zeros(2)
    keys = pygame.key.get_pressed()
    
    # Assuming a 2D action space for movement
    # Modify these mappings based on your environment's action space
    if keys[pygame.K_LEFT]:
        action[0] = -1.0
    if keys[pygame.K_RIGHT]:
        action[0] = 1.0
    if keys[pygame.K_UP]:
        action[1] = -1.0  # Usually up is negative in gym environments
    if keys[pygame.K_DOWN]:
        action[1] = 1.0
    
    return action

def reset_environment(env):
    rs = np.random.RandomState()
    state = np.array(
        [
            rs.randint(50, 450),
            rs.randint(50, 450),
            rs.randint(100, 400),
            rs.randint(100, 400),
            rs.randn() * 2 * np.pi - np.pi,
            300, 
            300,
            np.pi/4
        ],
        dtype=np.float64
    )
    obs, info = env.reset(options={"reset_to_state": state})
    return obs, info
def policy_rollout(dataset, env_name="gym_pusht/PushT-v0", obs_type="keypoints", render_mode="human"):
    # Load the demonstration fil

    env = gym.make(env_name, obs_type=obs_type, render_mode=render_mode)

    obs, info = reset_environment(env)
    #choose an arbitrary starting state
    total_reward = 0
    
    source_distribution= dataset['distribution']
    action_tesor = dataset['action']

    state_weight= 0.2
    transport=PolicyTransportation()
    method = Iterative_Locally_Weighted_Translations(num_iterations=30, rho=0.9, beta=0.9)
    # method = GPR(kernel=C(0.1) * RBF(length_scale=[0.1]) + WhiteKernel(0.0001))
    # method = BiJectiveNetwork(num_epochs=1000, num_blocks=4, num_hidden=50)
    transport.set_method(method=method, is_residual=False)
    for _ in range(1000):

        delta_x = get_keyboard_action()  

        if abs(delta_x[0])>0 or abs(delta_x[1])>0:
            action = obs['agent_pos'] + delta_x*5
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            pygame.time.Clock().tick(20)
            if terminated: 
                print("Episode terminated, resetting environment.")
                obs, info = reset_environment(env)
        else: 
            target_distribution = read_point_distribution(obs)
            target_distribution = np.expand_dims(target_distribution, axis=0)
            # Find the closest point between target_distribution and source_distribution along the batch dimension
            distances = np.linalg.norm(source_distribution - target_distribution, axis=2)
            weights = (1-state_weight)/(distances.shape[1]-1)*np.ones((1,distances.shape[1]-1))
            weights= np.append(weights, state_weight)  # Add weight for the agent state
            weights = np.expand_dims(weights, axis=0)
            weighted_distances = distances * weights
            distances = np.sum(weighted_distances, axis=1)  # Sum over the keypoints dimension
            closest_idx = np.argmin(distances)
            action_chunk = action_tesor[closest_idx]
            print(f"Closest index: {closest_idx}, Distance: {distances[closest_idx]}")

            transport.fit(source_distribution[closest_idx], target_distribution[0], do_scale=False, do_rotation=False)
            action_chunk=transport.transport(action_chunk, return_std=False)
            for action in action_chunk[:20]:
                delta_action = np.clip( action - obs['agent_pos'], -delta_action_lim, delta_action_lim)
                action = obs['agent_pos'] + delta_action
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward 
                env.render()  # Visualize each step

                pygame.time.Clock().tick(20)  # Limit to 60 FPS
                if terminated: 
                    print("Episode terminated, resetting environment.")
                    obs, info = reset_environment(env)
    
    print(f"Demonstration playback complete. Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.dataset, 'rb') as f:
        dataset = pickle.load(f)
    policy_rollout(dataset)