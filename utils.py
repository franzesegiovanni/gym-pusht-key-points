import numpy as np
import pickle
def read_point_distribution(observation):
    """
    Extracts the point distribution from the observation.
    Assumes the observation contains 'object_keypoints' and 'goal_keypoints'.
    """
    keypoint_object = observation['object_keypoints']
    keypoint_goal = observation['goal_keypoints']
    agent_state=observation['agent_pos']
    
    # Reshape keypoints to ensure they are in the correct format
    keypoint_object = np.array(keypoint_object).reshape(-1, 2)
    # print(keypoint_object.shape)
    keypoint_goal = np.array(keypoint_goal).reshape(-1, 2)
    # print(keypoint_goal.shape)
    agent_state= agent_state.reshape(-1,2)
    # print(agent_state.shape)
    # Combine object and goal keypoints
    distribution = np.vstack((keypoint_object, keypoint_goal,agent_state))
    # print(distribution.shape)

    
    return distribution

def save_dataset(dataset, name='dataset.pkl'):
    with open(name, 'wb') as f:
        pickle.dump(dataset, f)