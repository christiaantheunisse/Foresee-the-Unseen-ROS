# Custom message to send a trajectory which is a path with a velocity and time
nav_msgs/Path path                      # the list of waypoints
geometry_msgs/Twist[] velocities        # desired velocity in each waypoint
geometry_msgs/Accel[] accelerations     # acceleration can be used in the trajectory follower as an estimate