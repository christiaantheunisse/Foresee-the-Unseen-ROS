#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "odometry_node.hpp"
#include "odometry_constants.hpp"

namespace racing_bot
{
  namespace odometry
  {

    OdometryNode::OdometryNode(const double wheel_radius, const double ticks_per_rev, const double wheel_base)
        : rclcpp::Node(NODE_NAME),
          orientation_(0.0),
          position_x_(0.0),
          position_y_(0.0),
          linear_velocity_x_(0.0),
          linear_velocity_y_(0.0),
          angular_velocity_(0.0),
          left_ticks_(0),
          right_ticks_(0),
          previous_left_ticks_(0),
          previous_right_ticks_(0),
          last_time_(this->now()),
          current_time_(this->now()),
          wheel_radius_(wheel_radius),
          ticks_per_rev_(ticks_per_rev),
          wheel_base_(wheel_base)
    {
      left_subscription_ = this->create_subscription<std_msgs::msg::Int32>(LEFT_SUBSCRIBER_TOPIC, ENCODER_QUEUE_SIZE, std::bind(&OdometryNode::leftCallBack, this, std::placeholders::_1));
      right_subscription_ = this->create_subscription<std_msgs::msg::Int32>(RIGHT_SUBSCRIBER_TOPIC, ENCODER_QUEUE_SIZE, std::bind(&OdometryNode::rightCallBack, this, std::placeholders::_1));
      odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(PUBLISHER_TOPIC, ODOMETRY_QUEUE_SIZE);

      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    }

    void OdometryNode::leftCallBack(const std_msgs::msg::Int32 left_message)
    {
      left_ticks_ = left_message.data;
      updateOdometry();
    }

    void OdometryNode::rightCallBack(const std_msgs::msg::Int32 right_message)
    {
      right_ticks_ = right_message.data;
    }

    void OdometryNode::updateOdometry()
    {
      current_time_ = this->now();
      const double delta_time = (current_time_ - last_time_).seconds();

      calculateVelocities(delta_time);
      updatePosition(delta_time);
      publishOdometry();
    }

    void OdometryNode::calculateVelocities(const double delta_time)
    {
      const double left_ticks_delta = left_ticks_ - previous_left_ticks_;
      const double right_ticks_delta = right_ticks_ - previous_right_ticks_;

      const double wheel_distance_per_tick = (wheel_radius_ * M_PI) / ticks_per_rev_;
      const double left_angular_velocity = (left_ticks_delta * wheel_distance_per_tick) / delta_time;
      const double right_angular_velocity = (right_ticks_delta * wheel_distance_per_tick) / delta_time;

      linear_velocity_x_ = ((right_angular_velocity + left_angular_velocity) / 2);
      linear_velocity_y_ = 0.0;
      angular_velocity_ = ((right_angular_velocity - left_angular_velocity) / wheel_base_);
    }

    void OdometryNode::updatePosition(const double delta_time)
    {
      const double displacement_x = (linear_velocity_x_ * cos(orientation_)) * delta_time;
      const double displacement_y = (linear_velocity_x_ * sin(orientation_)) * delta_time;
      const double rotation_angle = angular_velocity_ * delta_time;

      position_x_ += displacement_x;
      position_y_ += displacement_y;
      orientation_ += rotation_angle;
    }

    void OdometryNode::publishOdometry()
    {
      // Create the odom message
      nav_msgs::msg::Odometry odom;
      odom.header.stamp = current_time_;
      odom.header.frame_id = FRAME_ID;
      odom.child_frame_id = CHILD_FRAME_ID;

      // Set the position
      geometry_msgs::msg::Quaternion odometry_quat = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), orientation_));
      odom.pose.pose.position.x = position_x_;
      odom.pose.pose.position.y = position_y_;
      odom.pose.pose.position.z = 0.0;
      odom.pose.pose.orientation = odometry_quat;

      // Set the velocity
      odom.twist.twist.linear.x = linear_velocity_x_;
      odom.twist.twist.linear.y = linear_velocity_y_;
      odom.twist.twist.angular.z = angular_velocity_;

      // Publish the odom message
      odometry_publisher_->publish(odom);

      // Create a transform
      geometry_msgs::msg::TransformStamped t;
      t.header = odom.header;
      t.child_frame_id = odom.child_frame_id;

      t.transform.translation.x = odom.pose.pose.position.x;
      t.transform.translation.y = odom.pose.pose.position.y;
      t.transform.translation.z = odom.pose.pose.position.z;

      t.transform.rotation.x = odom.pose.pose.orientation.x;
      t.transform.rotation.y = odom.pose.pose.orientation.y;
      t.transform.rotation.z = odom.pose.pose.orientation.z;
      t.transform.rotation.w = odom.pose.pose.orientation.w;

      // Broadcast the transform
      tf_broadcaster_->sendTransform(t);
      RCLCPP_INFO_ONCE(this->get_logger(), "Transform broadcasted");

      last_time_ = current_time_;
      previous_left_ticks_ = left_ticks_;
      previous_right_ticks_ = right_ticks_;
    }
  }
}