#ifndef ODOMNODE_H
#define ODOMNODE_H

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/pose_with_covariance.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace racing_bot
{
  namespace odometry
  {

    /**
     * @brief Constructs an Odometry Node object with the specified wheel radius, ticks per revolution, and wheelbase.
     *
     * This class represents an Odometry Node that calculates and publishes the odometry of a robot based on wheel encoder ticks.
     * The constructor initializes the Odometry Node with the given wheel radius, ticks per revolution, and wheelbase.
     * It sets up subscriptions to left and right wheel encoder topics, and publishes the calculated odometry on the "odom" topic.
     *
     * @param wheel_radius The radius of the wheels of the robot.
     * @param ticks_per_rev The number of encoder ticks per revolution of the wheels.
     * @param wheel_base The distance between the wheels of the robot.
     */
    class OdometryNode : public rclcpp::Node
    {
    public:
      OdometryNode();

    private:
      void leftCallBack(const std_msgs::msg::Int32 left_message);
      void rightCallBack(const std_msgs::msg::Int32 right_message);
      void updateOdometry();
      void publishOdometry();
      void calculateVelocities(const double delta_time);
      void updatePosition(const double delta_time);

      double orientation_, position_x_, position_y_, linear_velocity_x_, linear_velocity_y_, angular_velocity_;
      int32_t left_ticks_, right_ticks_, previous_left_ticks_, previous_right_ticks_;
      rclcpp::Time last_time_, current_time_;
      double wheel_radius_, ticks_per_rev_, wheel_base_;

      rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr left_subscription_;
      rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
      rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr right_subscription_;
      std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

      std::string odom_frame_;
      std::string base_frame_;
      bool do_broadcast_transform_;
      std::vector<double> pose_variances_;
      std::vector<double> twist_variances_;
    };
  }
}
#endif