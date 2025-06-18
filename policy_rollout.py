#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
import pickle
import os
import argparse
import gym_pusht
import pygame
from utils import read_point_distribution, save_dataset, load_dataset, plot_distribution, create_boxplot, save_video
from policy_transportation.transportation.transportation import PolicyTransportation
from policy_transportation.models.locally_weighted_translations import Iterative_Locally_Weighted_Translations
import seaborn as sns
from matplotlib import pyplot as plt
import cv2

delta_action_lim= 5.0
dataset_name = 'dataset_single.pkl'
control_frequency = 20  # Control frequency in Hz
limit_time = 120  # Limit time in seconds
number_samples =5
number_rollouts = 3
temperature = 1.2  # Temperature coefficient to regulate probability distribution
error_treshold = 20
display_distribution = False
use_relative_actions = False
use_diffeomorphism = True
hg_dagger = False
render_mode = "human"  # Options: "human", "rgb_array", or None
# render_mode = "rgb_array"  # Options: "human", "rgb_array", or Nones
reward_list = []
failure_list = []
timer_list = []


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

def samples_from_tranjectory(observation_traj, action_traj, action_chunch_leght=control_frequency):
    distribution_dim = 25  # Length of action chunks
    distribution_tensor=np.empty((0, distribution_dim, 2))  # Initialize an empty tensor for distributions
    action_tensor = np.empty((0, action_chunch_leght, 2))

    for i in range(len(observation_traj) - action_chunch_leght):
        distribution_tensor = np.append(distribution_tensor, observation_traj[i].reshape(1, distribution_dim, 2), axis=0)
        action_chunch =[]
        for j in range(action_chunch_leght):
            action_chunch.append(action_traj[i+j])
        action_tensor = np.append(action_tensor, np.array(action_chunch).reshape(1, action_chunch_leght, 2), axis=0)
    
    return distribution_tensor, action_tensor

def policy_rollout(dataset_name, env_name="gym_pusht/PushT-v0", obs_type="keypoints", render_mode="human"):
    # Load the demonstration fil
    dataset=load_dataset(dataset_name)
    env = gym.make(env_name, obs_type=obs_type, render_mode=render_mode)
    #choose an arbitrary starting state
    total_reward = 0
    print ("Loaded a dataset with dimensions:", dataset['distribution'].shape[0])
    source_distribution= dataset['distribution']
    action_tensor = dataset['action']

    state_weight= 0.3
    transport=PolicyTransportation()
    method = Iterative_Locally_Weighted_Translations(num_iterations=30, rho=0.9, beta=0.9)
    transport.set_method(method=method, is_residual=False)
    human_action_traj =[] 
    human_observsation_traj = []
    images= []
    timer = 0
    failure = 0
    success = 0
    for rollout in range(number_rollouts):
        print(f"Starting rollout {rollout + 1}/{number_rollouts}")
        obs, info = reset_environment(env)
        terminated = False

        while True:
            if terminated:
                break
            if render_mode == "human":
                delta_x = get_keyboard_action()  

            if timer > limit_time * control_frequency: 
                print("Time limit reached, resetting environment.")
                failure +=1
                reward_list.append(reward)
                failure_list.append(True)
                timer_list.append(timer/control_frequency)
                if render_mode == 'rgb_array':
                    video_name = f'video/rollout_video_{rollout}_failure.mp4'
                    save_video(images, video_name, control_frequency)      
                    images = []
                timer = 0
                break
            if hg_dagger and (abs(delta_x[0])>0 or abs(delta_x[1]))>0 and render_mode == "human":
                print("Human took control")
                if  min_distance < 10:
                    print("Deleted a conflicting point from the dataset at distance:", min_distance)
                    action_tensor = np.delete(action_tensor, closest_idx, axis=0)
                    source_distribution = np.delete(source_distribution, closest_idx, axis=0)
                    min_distance = np.inf
                    
                action = obs['agent_pos'] + delta_x*5
                obs, reward, terminated, truncated, info = env.step(action)

                env.render()
                observed_distribution = read_point_distribution(obs)
                human_action_traj.append(action)
                human_observation_traj.append(observed_distribution)

                pygame.time.Clock().tick(control_frequency)
                timer=0
                if terminated: 
                    print("Episode terminated, resetting environment.")
                    print(f"Total reward: {total_reward}")
                    break

            else: 
                if len(human_action_traj) >action_tensor.shape[1]:
                    incremental_distribution_tensor, incremental_action_tensor=samples_from_tranjectory(human_observsation_traj, human_action_traj)
                    human_action_traj =[] 
                    human_observation_traj = []
                    source_distribution = np.append(source_distribution, incremental_distribution_tensor, axis=0)            
                    action_tensor = np.append(action_tensor, incremental_action_tensor, axis=0)
                    dataset['distribution'] = source_distribution
                    dataset['action'] = action_tensor
                    print("New samples added to the dataset.")
                    print(f"Added distribution tensor shape: {incremental_distribution_tensor.shape}")
                    save_dataset(dataset, name=dataset_name)

                target_distribution = read_point_distribution(obs)
                target_distribution = np.expand_dims(target_distribution, axis=0)
                # Find the closest point between target_distribution and source_distribution along the batch dimension
                distances = np.linalg.norm(source_distribution - target_distribution, axis=2)
                weights = (1-state_weight)/(distances.shape[1]-1)*np.ones((1,distances.shape[1]-1))
                weights= np.append(weights, state_weight)  # Add weight for the agent state
                weights = np.expand_dims(weights, axis=0)
                weighted_distances = distances * weights
                distances = np.sum(weighted_distances, axis=1)  # Sum over the keypoints dimension

                # Find the smallest distances and pick a random on
                top_indices = np.argpartition(distances, number_samples)[:number_samples]
                # Create decreasing probabilities for the top indices
                probabilities = np.exp(-np.arange(len(top_indices)) / temperature)
                probabilities = probabilities / probabilities.sum()  # Normalize to sum to 1
                closest_idx = np.random.choice(top_indices, p=probabilities)

                min_distance = distances[closest_idx]
                action_chunk = action_tensor[closest_idx]

                if use_relative_actions:
                    action_chunk_relative = action_chunk - action_chunk[0]  # Relative to the first action in the chunk
                    action_chunk = action_chunk_relative + obs['agent_pos']  # Add the current agent position to the relative actionss
                
                # print(f"Closest index: {closest_idx}, Distance: {distances[closest_idx]}")
                if use_diffeomorphism:
                    chosen_source_distribution = source_distribution[closest_idx].copy()
                    chosen_target_distribution = target_distribution[0].copy()
                    transport.fit(chosen_source_distribution, chosen_target_distribution, do_scale=False, do_rotation=False)
                    source_distribution_tranported = transport.transport(chosen_source_distribution, return_std=False)
                    error = np.linalg.norm(source_distribution_tranported - chosen_target_distribution)
                    if display_distribution:
                        action_chunk_transported = transport.transport(action_chunk, return_std=False)
                        plot_distribution(chosen_source_distribution, chosen_target_distribution, action_chunk, action_chunk_transported)

                    if error > error_treshold:
                        # print(f"Error in transportation: {error}, removing GOAL keypoints")
                        chosen_source_distribution = np.delete(chosen_source_distribution, np.s_[12:24], axis=0)
                        chosen_target_distribution = np.delete(chosen_target_distribution, np.s_[12:24], axis=0)
                        transport.fit(chosen_source_distribution, chosen_target_distribution, do_scale=False, do_rotation=False)

                    action_chunk_transported=transport.transport(action_chunk, return_std=False)
                    source_distribution_tranported = transport.transport(chosen_source_distribution, return_std=False)
                    error = np.linalg.norm(source_distribution_tranported - chosen_target_distribution)
                else:
                    action_chunk_transported = action_chunk.copy()

                for action in action_chunk_transported:
                    delta_action = np.clip( action - obs['agent_pos'], -delta_action_lim, delta_action_lim)
                    action = obs['agent_pos'] + delta_action
                    obs, reward, terminated, truncated, info = env.step(action)
                    key_points = read_point_distribution(obs)
                    reward = -np.linalg.norm(key_points[:12, :]- key_points[12:24, :], axis=1).mean()  # Reward based on the distance between object and goal keypoints
                    img  = env.render()
                    images.append(img)  # Visualize each step
                    timer+=1
                    pygame.time.Clock().tick(control_frequency)  # Limit to 60 FPS
                    if terminated: 
                        print("Episode terminated, resetting environment.")
                        print(f"Final reward: {reward}")
                        reward_list.append(reward)
                        failure_list.append(False)
                        timer_list.append(timer/control_frequency)
                        # obs, info = reset_environment(env)
                        timer=0
                        success +=1
                        if render_mode == 'rgb_array':
                            video_name = f'video/rollout_video_{rollout}_success.mp4'
                            save_video(images, video_name, control_frequency)      
                            images = []
                        break

    print(f"Total successes: {success}, Total failures: {failure}")
    fig, ax = create_boxplot(reward_list, failure_list, timer_list)
    # Save reward_list and failure_list to pickle file
    title = f'use_relative_actions={use_relative_actions}_use_diffeomorphism={use_diffeomorphism}_number_rollouts={number_rollouts}_number_samples={number_samples}'

    pickle_data = {
        'reward_list': reward_list,
        'failure_list': failure_list,
        'timer_list': timer_list,
    }
    with open(f'results_{title}.pkl', 'wb') as f:
        pickle.dump(pickle_data, f)
    plt.title(title)
    fig.savefig(f'box_plot_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay a saved demonstration")
    parser.add_argument("--dataset", type=str, default='dataset_single.pkl',)
    args = parser.parse_args()
    
    policy_rollout(dataset_name=dataset_name, render_mode=render_mode)