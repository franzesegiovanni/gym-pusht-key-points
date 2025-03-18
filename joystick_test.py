import pygame

# Initialize pygame and joystick
pygame.init()
pygame.joystick.init()

# Check if a joystick is connected
if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Connected to joystick: {joystick.get_name()}")

try:
    while True:
        pygame.event.pump()  # Process events
        
        # Read axis values
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        axes = [round(axis, 2) for axis in axes]
        axes= axes[0:2]
        print(f"Axes: {axes}")

        # Read button states
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        # print(f"Buttons: {buttons}")

        # Read hat (D-pad) states
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
        # print(f"Hats: {hats}")

        pygame.time.delay(100)  # Delay to reduce CPU usage

except KeyboardInterrupt:
    print("Exiting...")
    pygame.quit()
