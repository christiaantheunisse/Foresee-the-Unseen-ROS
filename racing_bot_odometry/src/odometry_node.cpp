#include "odometry_node.hpp"
namespace racing_bot {
    namespace odometry {

        OdometryNode::OdometryNode()
            : rclcpp::Node("odom_node"),
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
            current_time_(this->now()) {
            // Declare the parameters with a default value
            this->declare_parameter("odom_frame", "odom");
            this->declare_parameter("base_frame", "base_link");
            this->declare_parameter("odometry_topic", "odom");
            this->declare_parameter("odometry_queue_size", 5);
            this->declare_parameter("encoder_queue_size", 1);
            this->declare_parameter("left_encoder_topic", "left_wheel");
            this->declare_parameter("right_encoder_topic", "right_wheel");
            this->declare_parameter("wheel_radius", 0.033);
            this->declare_parameter("ticks_per_rev", 1920.);
            this->declare_parameter("wheel_base", 0.153);
            this->declare_parameter("do_broadcast_transform", false);
            this->declare_parameter("pose_variances", std::vector<double>({ 0., 0., 0., 0., 0., 0. }));
            this->declare_parameter("twist_variances", std::vector<double>({ 0., 0., 0., 0., 0., 0. }));

            // Set the variables by reading the parameters -> No callback needed
            auto odometry_topic = this->get_parameter("odometry_topic").as_string();            // local
            auto odometry_queue_size = this->get_parameter("odometry_queue_size").as_int();     // local
            auto encoder_queue_size = this->get_parameter("encoder_queue_size").as_int();       // local
            auto left_encoder_topic = this->get_parameter("left_encoder_topic").as_string();    // local
            auto right_encoder_topic = this->get_parameter("right_encoder_topic").as_string();  // local
            odom_frame_ = this->get_parameter("odom_frame").as_string();
            base_frame_ = this->get_parameter("base_frame").as_string();
            wheel_radius_ = this->get_parameter("wheel_radius").as_double();
            ticks_per_rev_ = this->get_parameter("ticks_per_rev").as_double();
            wheel_base_ = this->get_parameter("wheel_base").as_double();
            do_broadcast_transform_ = this->get_parameter("do_broadcast_transform").as_bool();
            pose_variances_ = this->get_parameter("pose_variances").as_double_array();
            twist_variances_ = this->get_parameter("twist_variances").as_double_array();

            left_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
                left_encoder_topic, encoder_queue_size, std::bind(&OdometryNode::leftCallBack, this, std::placeholders::_1));
            right_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
                right_encoder_topic, encoder_queue_size, std::bind(&OdometryNode::rightCallBack, this, std::placeholders::_1));
            odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(odometry_topic, odometry_queue_size);

            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        void OdometryNode::leftCallBack(const std_msgs::msg::Int32 left_message) {
            left_ticks_ = left_message.data;
            updateOdometry();
        }

        void OdometryNode::rightCallBack(const std_msgs::msg::Int32 right_message) { right_ticks_ = right_message.data; }

        void OdometryNode::updateOdometry() {
            current_time_ = this->now();
            const double delta_time = (current_time_ - last_time_).seconds();
            // RCLCPP_INFO(this->get_logger(), "The delta_time: %f", delta_time);
            if (delta_time < 1e-4) {
                // RCLCPP_INFO(this->get_logger(), "No odom data published, since the delta_time is %f", delta_time);
                return;
            }
            else if (delta_time > 1.) {
                // RCLCPP_INFO(this->get_logger(), "No odom data published, since the delta_time is %f", delta_time);
                last_time_ = current_time_;
                previous_left_ticks_ = left_ticks_;
                previous_right_ticks_ = right_ticks_;
                return;
            }
            calculateVelocities(delta_time);
            updatePosition(delta_time);
            publishOdometry();
        }

        void OdometryNode::calculateVelocities(const double delta_time) {
            const double left_ticks_delta = left_ticks_ - previous_left_ticks_;
            const double right_ticks_delta = right_ticks_ - previous_right_ticks_;

            const double wheel_distance_per_tick = (wheel_radius_ * M_PI) / ticks_per_rev_;
            const double left_angular_velocity = (left_ticks_delta * wheel_distance_per_tick) / delta_time;
            const double right_angular_velocity = (right_ticks_delta * wheel_distance_per_tick) / delta_time;

            linear_velocity_x_ = ((right_angular_velocity + left_angular_velocity) / 2);
            linear_velocity_y_ = 0.0;
            angular_velocity_ = ((right_angular_velocity - left_angular_velocity) / wheel_base_);
        }

        void OdometryNode::updatePosition(const double delta_time) {
            const double displacement_x = (linear_velocity_x_ * cos(orientation_)) * delta_time;
            const double displacement_y = (linear_velocity_x_ * sin(orientation_)) * delta_time;
            const double rotation_angle = angular_velocity_ * delta_time;

            position_x_ += displacement_x;
            position_y_ += displacement_y;
            orientation_ += rotation_angle;
        }

        void OdometryNode::publishOdometry() {
            // Create the odom message
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = current_time_;
            odom.header.frame_id = odom_frame_;
            odom.child_frame_id = base_frame_;

            // Set the pose
            geometry_msgs::msg::PoseWithCovariance pose_msg;
            pose_msg.pose.position.x = position_x_;
            pose_msg.pose.position.y = position_y_;
            pose_msg.pose.position.z = 0.0;

            geometry_msgs::msg::Quaternion odometry_quat = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), orientation_));
            pose_msg.pose.orientation = odometry_quat;

            pose_msg.covariance[0] = pose_variances_[0];
            pose_msg.covariance[7] = pose_variances_[1];
            pose_msg.covariance[14] = pose_variances_[2];
            pose_msg.covariance[21] = pose_variances_[3];
            pose_msg.covariance[28] = pose_variances_[4];
            pose_msg.covariance[35] = pose_variances_[5];

            odom.pose = pose_msg;

            // Set the twist
            geometry_msgs::msg::TwistWithCovariance twist_msg;
            twist_msg.twist.linear.x = linear_velocity_x_;
            twist_msg.twist.linear.y = linear_velocity_y_;
            twist_msg.twist.angular.z = angular_velocity_;

            twist_msg.covariance[0] = twist_variances_[0];
            twist_msg.covariance[7] = twist_variances_[1];
            twist_msg.covariance[14] = twist_variances_[2];
            twist_msg.covariance[21] = twist_variances_[3];
            twist_msg.covariance[28] = twist_variances_[4];
            twist_msg.covariance[35] = twist_variances_[5];

            odom.twist = twist_msg;

            // odom.pose.pose.position.x = position_x_;
            // odom.pose.pose.position.y = position_y_;
            // odom.pose.pose.position.z = 0.0;
            // odom.pose.pose.orientation = odometry_quat;

            // Set the velocity
            // odom.twist.twist.linear.x = linear_velocity_x_;
            // odom.twist.twist.linear.y = linear_velocity_y_;
            // odom.twist.twist.angular.z = angular_velocity_;

            // Set covariance
            // std_msgs::msg::Float64MultiArray covariance;
            // covariance.data.resize(36);
            // covariance.data[0] = 0.1;
            // covariance.data[7] = 0.1;
            // covariance.data[14] = 0.1;
            // covariance.data[21] = 0.1;
            // covariance.data[28] = 0.1;
            // covariance.data[35] = 0.1;

            // Publish the odom message
            odometry_publisher_->publish(odom);

            if (do_broadcast_transform_) {
                RCLCPP_INFO_ONCE(this->get_logger(), "Transform broadcasted");

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
            }

            last_time_ = current_time_;
            previous_left_ticks_ = left_ticks_;
            previous_right_ticks_ = right_ticks_;
        }
    }  // namespace odometry
}  // namespace racing_bot