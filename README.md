# Foresee the Unseen on a real robot

The main goal is to transfer the algorithm from the paper *Foresee The Unseen: Sequential Reasoning about
Hidden Obstacles for Safe Driving* ([IEEE](https://ieeexplore.ieee.org/document/9827171) / [open access link](https://www.diva-portal.org/smash/get/diva2:1635726/FULLTEXT01.pdf)) to a real robot, take the necessary measures against sensor noises and delays, and to verify its performance in the real world. The goal of this algorithm, which is further referred to as **Foresee the Unseen** is to safely, but not overconservatively deal with occlusions encountered by autonomous vehicles. The algorithm is implemented in ROS and this repository also contains all the code to get control the robot. The picture below shows the robots on which the software is usable.

The 'autonomous vehicle'             |  The obstacle cars
:-------------------------:|:-------------------------:
![](media/racing_bot.jpg)  |  ![](media/obstacle_robots.jpg)

The video below shows a preliminairy result of the implemented algorithm.

![Algorithm .gif](media/rviz_visualization.gif)

## Description of the packages
A short description of all the packages in this repo is given below. The code was written as part of my graduation internship at ALTEN and most of the code used for the low-level control of the robot was written by Catuja Smit.


### `foresee_the_unseen`

This code package implements the Foresee the Unseen algorithm and contains a few nodes, which are all related to the algorithm or testing of it:

- `fov_node.py`: This node constructs the field of view (FOV) used in the algorithm. It receives the `LaserScan` messages from the Lidar sensor and outputs a `StampedPolygon` describing the FOV. The procedure is further described in the [thesis paper]().
- `planner_node.py`: The most important node that connects the code from the algorithm to the ROS part. Initially the goal was to also setup a simple simulation, so the code for the algorithm (in the folder `lib`) does import any ROS packages.
- `obstacle_trajectories.py`: This node publishes the trajectories for all the obstacle cars used in the experiments, both simulated and real cars.
- `logging_node.py`/`topics_to_disk_node.py`: Are basically the same and can subscribe to multiple topics and write the messages to disk.


### `racing_bot_bringup`

This package only contains launch files (`launch`) to launch the nodes and the configuration files (`config`) for `slam_toolbox` and `robot_localization`, which are used for respectively the localization and Kalman filters. 

I wrote a tutorial on how to use these packages for a course in the Master Robotics at the Technical University of Delft, which can be found [here](https://github.com/christiaantheunisse/localization-with-ROS):

- `slam_toolbox`: Lidar-based SLAM is used for the localization of the robot. Depending on the update rate, the SLAM node has to be run on a laptop or can run on a Raspberry Pi 4. The 'autonomous vehicle' robot uses an update rate of 0.5 s, so the SLAM node has to run on the laptop, but the obstacle robots run the SLAM node onboard with a rate of 2 s.
- `robot_localization`: Two EKFs are implemented which rely on the Lidar-based SLAM, the odometry and an IMU and is able to provide a stable position and velocity output (see the tutorial for more information).

### `racing_bot_controller`

This package is partly written by Catuja Smit and has two nodes. One node to convert ROS Twist messages (linear and angular velocity) to motor pwm values for a differential drive robot (DDR) which are published as an array of integers. However, this node is not updated and does not work correctly. I hacked a simple node together to send these messages based on keyboard inputs for testing purposes.

### `racing_bot_encoder`

This package (also written by Catuja Smit) contains one node that reads the values from the encoders and publishes these on a topic. I changed the published message type to a custom message described in `racing_bot_interfaces`, namely `EncoderValues.msg`. Both encoder values are now published at the same time with a time stamp to improve the velocity calculation in the odometry node.

### `racing_bot_hat`

This package (also written by Catuja Smit) contains one node that controls the motors through the Adafruit motor driver hat that sits on top of the Raspberry Pi. I have not made any significant changes to this node.

Install the required `pigpiod` library on the Raspberry Pi ([installation instructions](https://abyz.me.uk/rpi/pigpio/download.html)) and make sure to run `sudo pigpiod` after every reboot of the Pi.

### `racing_bot_imu`

This packages (also written by Catuja Smit) contains one node that reads the values from an IMU and publishes them. The MadgwickAHRS algorithm is used to calculate the orientation. I added some missing dependencies files for the algorithm that were missing in the version I got and changes to frames to make them work with the IMU I got (my IMU, like a lot of IMUs outputs the readings in a lefthanded coordinate frame.)

### `racing_bot_interfaces`

Contains the custom ROS message types, which are:

- `EncoderValues`: A message to send to encoder values (integers) at the same time with a timestamp.
- `ProjectedOccludedArea`: Simple a stamped float that contains the total occluded area, which was used for the analysis of the algorithm. The ROS convention is to always define custom message times because a Float or StampedFloat does not provide enough context and should only be used as building block for other messages.
- `Trajectory`: To describe a list of states consisting of a timestamp, pose, velocity and optionally acceleration.

### `racing_bot_odometry`

The single node in this package calculates the position and velocity based on the encoder readings. The position calculations take a more geometrical approach now and are independent of the velocity. The calculate position is not used in the final approach, but seems to work fine. The velocity calculations are now based on multiple past measurement to reduce the noise and take the measurement time into account.

### `racing_bot_scan_sim`

This package contains a node that adds simulated vehicles or a static obstacle to the `LaserScan` message. The position of the obstacles are obtained from multiple topics. The vehicles are simulated as simple planes perpendicular to the mobile robot, but the static obstacle is defined as a polygon and is added with the right shape to the FOV.

### `racing_bot_simulation`

This package has to nodes and allows to run all the code without the actual robot. It was used to design scenarios for the experiments with having to do a lot of iterations with the real robot.

- `odometry_sensor_node.py`: This node publishes tje odometry by linearly interpolating the trajectory.
- `scan_sensor_node.py`: This node publishes a scan messages with only maximum range measurements.


### `racing_bot_trajectory_follower`

I wrote this package from scratch and it contains a single node which enables the robot to follow a trajectory. The node uses Stanley steering to control the steering angle which is converted to a desired angular velocity around the z-axis and tracked with PD-controller. A P-controller is used for the linear velocity. The idea and initial implementation were taken from [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics?tab=readme-ov-file#stanley-control), although a lot of changes were made to make it applicable to differential drive robots and make it track a list of stamped states. 

The input message type is defined in `racing_bot_interfaces` in `Trajectory.msg`, which contains a list of states. Every state consists of a position, velocity, acceleration and time stamp. However, the time stamp and acceleration are optional. The time stamp is necessary when the goal states of the robot are time dependant and the accelerations can be provide as an initial guess for the necessary change in the PWM signal. However, the mode that uses the accelerations makes the robot velocity unstable and wiggly!

Since the approach was changed and improved over time, a lot of code and functions accumulated over time, which might not all work with anymore, so together it's a big mess. However, the controllers for the angular and linear velocity work truely well, so I think I will move these to a separate node. This will allow someone else to write a better trajectory follower on top of these controllers.


