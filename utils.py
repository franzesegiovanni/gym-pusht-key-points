import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
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

def load_dataset(name='dataset.pkl'):
    with open(name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def create_boxplot(reward_list, failure_list, timer_list):

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot: Reward violin plot
    parts1 = ax1.violinplot([reward_list], positions=[0], widths=0.6, showmeans=True, showmedians=True)
    
    # Customize violin plot colors for rewards
    for pc in parts1['bodies']:
        pc.set_facecolor('lightblue')
        pc.set_alpha(0.7)
    
    # Add scatter plot overlay for rewards
    y_positions = np.zeros(len(reward_list))  # All points at y=0 for horizontal alignment
    
    # Plot failures as 'x' and successes as circles for rewards
    for i, (reward, is_failure) in enumerate(zip(reward_list, failure_list)):
        if is_failure:
            ax1.scatter(y_positions[i], reward, marker='x', color='red', s=100, alpha=0.8, label='Failure' if i == 0 or not any(failure_list[:i]) else "")
        else:
            ax1.scatter(y_positions[i], reward, marker='o', color='green', s=50, alpha=0.8, label='Success' if i == 0 or all(failure_list[:i+1]) else "")
    
    # Customize the first subplot
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Distribution')
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Timer violin plot
    parts2 = ax2.violinplot([timer_list], positions=[0], widths=0.6, showmeans=True, showmedians=True)
    
    # Customize violin plot colors for timer
    for pc in parts2['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    # Add scatter plot overlay for timer
    y_positions_timer = np.zeros(len(timer_list))  # All points at y=0 for horizontal alignment
    
    # Plot failures as 'x' and successes as circles for timer
    for i, (timer, is_failure) in enumerate(zip(timer_list, failure_list)):
        if is_failure:
            ax2.scatter(y_positions_timer[i], timer, marker='x', color='red', s=100, alpha=0.8, label='Failure' if i == 0 or not any(failure_list[:i]) else "")
        else:
            ax2.scatter(y_positions_timer[i], timer, marker='o', color='green', s=50, alpha=0.8, label='Success' if i == 0 or all(failure_list[:i+1]) else "")
    
    # Customize the second subplot
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylabel('Timer')
    ax2.set_title('Timer Distribution')
    ax2.set_xticks([])
    ax2.grid(True, alpha=0.3)
    
    # Add legend to the first subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles1:
        ax1.legend()
    
    # Add legend to the second subplot
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend()
    
    plt.tight_layout()
    return fig, (ax1, ax2)

def plot_distribution(source_distribution, target_distribution, action_chunk, action_chunk_transported):
    
    # Initialize plots if they don't exist
    if not hasattr(plot_distribution, 'fig1'):
        plot_distribution.fig1, plot_distribution.ax1 = plt.subplots(1, 1, figsize=(12, 5))
        plt.ion()  # Turn on interactive mode

    # Clear previous plots
    plot_distribution.ax1.clear()
    # plot_distribution.ax3.clear()

    # Figure 1: Source distribution and action chunk
    plot_distribution.ax1.scatter(source_distribution[:12, 0], source_distribution[:12, 1], 
                c='blue', alpha=0.6)
    
    plot_distribution.ax1.scatter(source_distribution[12:24, 0], source_distribution[12:24, 1], 
                c='green', alpha=0.6)
    plot_distribution.ax1.scatter(source_distribution[24, 0], source_distribution[24, 1], 
                c='red', alpha=0.6, marker='*', s=400)
    plot_distribution.ax1.scatter(target_distribution[24, 0], target_distribution[24, 1],
                c='black', alpha=0.6, marker='*', s=400, label='Agent Position')
    plot_distribution.ax1.plot([source_distribution[24, 0], target_distribution[24, 0]], 
                [source_distribution[24, 1], target_distribution[24, 1]], 
                'black', alpha=1, linewidth=2)
    
    plot_distribution.ax1.scatter(target_distribution[:12, 0], target_distribution[:12, 1], 
                c='purple', alpha=0.6, label='Target Distribution')
    # Plot connection lines between source and target distributions
    for i in range(12):
        plot_distribution.ax1.plot([source_distribution[i, 0], target_distribution[i, 0]], 
                                 [source_distribution[i, 1], target_distribution[i, 1]], 
                                 'gray', alpha=1, linewidth=2)
    for i in range(len(action_chunk)):
       plot_distribution.ax1.plot([action_chunk[i, 0], action_chunk_transported[i, 0]],
                                 [action_chunk[i, 1], action_chunk_transported[i, 1]],
                                 'orange', alpha=0.6, linewidth=2) 
    plot_distribution.ax1.set_title('Source Distribution (Closest Match)')
    plot_distribution.ax1.set_xlim(0, 500)
    plot_distribution.ax1.set_ylim(0, 500)
    plot_distribution.ax1.set_aspect('equal', adjustable='box')
    plot_distribution.ax1.set_xlabel('X')
    plot_distribution.ax1.set_ylabel('Y')
    plot_distribution.ax1.invert_yaxis()
    plot_distribution.ax1.legend()
    plot_distribution.ax1.grid(True)

    plot_distribution.ax1.scatter(action_chunk[:, 0], action_chunk[:, 1], 
                c='red', alpha=0.6, label='Action Chunk')

    plot_distribution.fig1.canvas.draw()
    plot_distribution.fig1.canvas.flush_events()
    
    plt.pause(0.01)

def save_video(images, video_name, control_frequency):
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, control_frequency, (width, height))
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()
    print(f"Video saved as {video_name}")