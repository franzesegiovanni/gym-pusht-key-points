import os
import glob
import gym
import pickle
import numpy as np

# Directory containing demonstration pickle files
demo_dir = "demonstration"

#list all the pickle files in the directory
demo_files = glob.glob(os.path.join(demo_dir, "*.pkl"))
distribution_dim = 25
distribution_tensor=np.empty((0, distribution_dim, 2))  # Initialize an empty tensor for distributions
action_chunch_leght=60  # Length of action chunks
action_tensor = np.empty((0, action_chunch_leght, 2))
for demo_path in demo_files:

    with open(demo_path, 'rb') as f:
        demo_data = pickle.load(f)

    for i in range(0,len(demo_data)-action_chunch_leght, 2):
    # reset to a specific state
        agent_state=demo_data[i]['observation']['agent_pos']
        keypoint_object=demo_data[i]['observation']['object_keypoints']
        keypoint_goal = demo_data[i]['observation']['goal_keypoints']
        keypoint_object = np.array(keypoint_object).reshape(-1, 2)
        keypoint_goal = np.array(keypoint_goal).reshape(-1, 2)
        distribution = np.vstack((keypoint_object, keypoint_goal,agent_state))
        distribution_tensor = np.append(distribution_tensor, distribution.reshape(1, distribution_dim, 2), axis=0)
        action_chunch =[]
        for j in range(action_chunch_leght):
            action_chunch.append(demo_data[i+j]['action'])
        action_tensor = np.append(action_tensor, np.array(action_chunch).reshape(1, action_chunch_leght, 2), axis=0)


dataset ={}
dataset['distribution']= distribution_tensor
dataset['action']= action_tensor
print("Dataset created with shapes:")
print(f"Distribution tensor shape: {dataset['distribution'].shape}")
print(f"Action tensor shape: {dataset['action'].shape}")
#save the object dataset as pickle

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
