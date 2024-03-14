#ifndef ODOMNODE_H
#define ODOMNODE_H

#include <cmath>
#include <deque>
#include <tuple>

#include "geometry_msgs/msg/pose_with_covariance.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "racing_bot_interfaces/msg/encoder_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace racing_bot {
    namespace odometry {

        struct deltaPose {
            rclcpp::Time stamp;
            double dt;
            double dx;
            double dy;
            double dtheta;
        };

        /**
         * @brief Constructs an Odometry Node object with the specified wheel radius, ticks per revolution, and
         * wheelbase.
         *
         * This class represents an Odometry Node that calculates and publishes the odometry of a robot based on wheel
         * encoder ticks. The constructor initializes the Odometry Node with the given wheel radius, ticks per
         * revolution, and wheelbase. It sets up subscriptions to left and right wheel encoder topics, and publishes the
         * calculated odometry on the "odom" topic.
         *
         * @param wheel_radius The radius of the wheels of the robot.
         * @param ticks_per_rev The number of encoder ticks per revolution of the wheels.
         * @param wheel_base The distance between the wheels of the robot.
         */
        class OdometryNode : public rclcpp::Node {
           public:
            OdometryNode();

           private:
            // void leftCallBack(const std_msgs::msg::Int32 left_message);
            // void rightCallBack(const std_msgs::msg::Int32 right_message);
            void EncoderCallback(const racing_bot_interfaces::msg::EncoderValues msg);
            void updatePose(racing_bot_interfaces::msg::EncoderValues msg);
            void calculateSpeed();
            void publishOdometry();

            std::vector<float> CalculatePoseVariances(float d_distance, float d_angle);

            double position_x_, position_y_, orientation_;
            double linear_speed_, angular_speed_;
            bool first_update = true;

            racing_bot_interfaces::msg::EncoderValues prev_msg;
            std::deque<deltaPose> pose_changes;
            double wheel_radius_, ticks_per_rev_, wheel_base_;

            rclcpp::Subscription<racing_bot_interfaces::msg::EncoderValues>::SharedPtr encoder_subscription_;
            rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
            std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

            double max_lookback_t_;
            std::string odom_frame_, base_frame_;
            bool do_broadcast_transform_;
            std::vector<double> pose_variances_, twist_variances_;
            double linear_var_, angular_var_;
        };
    }  // namespace odometry
}  // namespace racing_bot
#endif