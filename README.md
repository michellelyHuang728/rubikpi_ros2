# Waypoint Navigation with ROS 2


## Code and Reproduction

### Requirements

- Ubuntu with ROS 2 Jazzy
- Rubik Pi 3
- WAVE ROVER robot
- Python 3
- ROS 2 `robot_control` package
- Serial connection to the motor controller

### Step 1: Clone the Repository

```bash
git clone https://github.com/michellelyHuang728/rubikpi_ros2.git
cd rubikpi_ros2
```

### Step 2: Build the ROS 2 Package

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_control --symlink-install
source install/setup.bash
```

### Step 3: Start Motor Control

Open Terminal 1:

```bash
cd ~/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run robot_control motor_control
```

The motor controller establishes a serial connection with the robot and subscribes to `/motor_commands`.

### Step 4: Run Waypoint Navigation

Open Terminal 2:

```bash
cd ~/rubikpi_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run robot_control waypoint_controller
```

The robot will execute the waypoint sequence loaded from `waypoints.txt`.

### Step 5: Stop the Robot

Press `Ctrl+C` in the waypoint controller terminal to stop the navigation process.

Ensure the robot is stationary before disconnecting power.

