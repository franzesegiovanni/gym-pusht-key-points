import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
import time
import json
import argparse

class DemonstrationRecorder:
    def __init__(self, env_name="gym_pusht/PushT-v0", obs_type="keypoints", 
                 render_mode="human", input_device="keyboard"):
        """
        Initialize the demonstration recorder
        """
        self.env = gym.make(env_name, obs_type=obs_type, render_mode=render_mode)
        self.input_device = input_device
        self.demonstration = []
        self.action_dim = self.env.action_space.shape[0]
        self.running = False
        self.current_episode = []
        
    def setup_input(self):
        """Initialize pygame for input handling"""
        pygame.init()
        # Need a small window for input capture
        self.screen = pygame.display.set_mode((320, 240))
        pygame.display.set_caption("Demo Recorder (Keep this window focused!)")
        
        if self.input_device == "joystick":
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Using joystick: {self.joystick.get_name()}")
            else:
                print("No joystick detected, falling back to keyboard")
                self.input_device = "keyboard"
    
    def get_keyboard_action(self):
        """Convert keyboard input to action"""
        action = np.zeros(self.action_dim)
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
    
    def get_joystick_action(self):
        """Convert joystick input to action"""
        action = np.zeros(self.action_dim)
        
        # Map joystick axes to action dimensions
        if self.action_dim >= 2:
            action[0] = self.joystick.get_axis(0)  # Left stick horizontal
            action[1] = -self.joystick.get_axis(1)  # Left stick vertical (inverted)
        
        return action
    
    def process_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    print("Resetting environment")
                    self.save_episode()
                    self.env.reset()
                    
    def save_episode(self):
        """Save the current episode if it has steps"""
        if len(self.current_episode) > 0:
            self.demonstration.append(self.current_episode)
            print(f"Episode {len(self.demonstration)} saved with {len(self.current_episode)} steps")
            self.current_episode = []
    
    def save_demonstration(self, filepath=None):
        """Save the recorded demonstration to a file"""
        # Make sure any active episode is saved
        self.save_episode()
        
        if not filepath:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filepath = f"demonstration_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.demonstration, f)
        
        print(f"Demonstration with {len(self.demonstration)} episodes saved to {filepath}")
    
    def record(self, max_steps=10000):
        """Main recording loop"""
        self.setup_input()
        
        obs, _ = self.env.reset()  # Ignore info
        self.running = True
        step = 0
        
        print("\n===== Demonstration Recorder =====")
        print("Controls:")
        print("  Arrow keys: Move the pusher (keyboard mode)")
        print("  R: Reset and start a new episode")
        print("  ESC: Stop recording")
        print("================================\n")
        
        while self.running and step < max_steps:
            self.env.render()  # Just render without assigning unused variable
            self.process_events()
            
            # Get action based on input device
            if self.input_device == "keyboard":
                delta_x = self.get_keyboard_action()
                print("delta_x",delta_x)
                state= self.env.get_obs()["agent_pos"]
                print("state",state)    
                action = self.env.get_obs()["agent_pos"] + delta_x*10
            else:
                action = self.get_joystick_action()
            
            # Take a step in the environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)  # Ignore info
            
            # Record step data
            step_data = {
                "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
                "action": action.tolist(),
                "reward": float(reward),
                "next_observation": next_obs.tolist() if hasattr(next_obs, 'tolist') else next_obs
            }
            self.current_episode.append(step_data)
            
            obs = next_obs
            step += 1
            
            if terminated or truncated:
                self.save_episode()
                obs, _ = self.env.reset()  # Ignore info using underscore
                        
            # Use a consistent frame rate to prevent flickering
            pygame.time.Clock().tick(60)  # Limit to 60 FPS
        
        # Clean up
        self.env.close()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Record demonstrations for gym-pusht')
    parser.add_argument('--device', type=str, default='keyboard', choices=['keyboard', 'joystick'],
                        help='Input device (keyboard or joystick)')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--obs-type', type=str, default='keypoints', help='Observation type')
    args = parser.parse_args()
    
    recorder = DemonstrationRecorder(
        env_name="gym_pusht/PushT-v0",  # Make sure this environment ID is correct
        render_mode="human",
        input_device=args.device,
        obs_type=args.obs_type
    )
    
    recorder.record()
    recorder.save_demonstration(args.output)

if __name__ == "__main__":
    main()