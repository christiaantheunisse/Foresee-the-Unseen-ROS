#include "odometry_node.hpp"
namespace racing_bot {
    namespace odometry {

        OdometryNode::OdometryNode()
            : rclcpp::Node("odom_node"),
              orientation_(0.),
              position_x_(0.),
              position_y_(0.),
              linear_speed_(0.),
              angular_speed_(0.) {
            // Declare the parameters with a default value
            this->declare_parameter("odom_frame", "odom");
            this->declare_parameter("base_frame", "base_link");
            this->declare_parameter("odometry_topic", "odom");
            this->declare_parameter("odometry_queue_size", 5);
            this->declare_parameter("encoder_queue_size", 1);
            this->declare_parameter("encoder_topic", "wheel_encoders");
            this->declare_parameter("wheel_radius", 0.032);
            this->declare_parameter("ticks_per_rev", 3840.);
            this->declare_parameter("wheel_base", 0.145);
            this->declare_parameter("do_broadcast_transform", false);
            this->declare_parameter("pose_variances", std::vector<double>({0., 0., 0., 0., 0., 0.}));
            this->declare_parameter("twist_variances", std::vector<double>({0., 0., 0., 0., 0., 0.}));
            this->declare_parameter("max_history_time", 0.15);
            this->declare_parameter("linear_var", 0.00642);
            this->declare_parameter("angular_var", 0.01610);

            // Set the variables by reading the parameters -> No callback needed
            auto odometry_topic = this->get_parameter("odometry_topic").as_string();         // local
            auto odometry_queue_size = this->get_parameter("odometry_queue_size").as_int();  // local
            auto encoder_queue_size = this->get_parameter("encoder_queue_size").as_int();    // local
            auto encoder_topic = this->get_parameter("encoder_topic").as_string();           // local
            odom_frame_ = this->get_parameter("odom_frame").as_string();
            base_frame_ = this->get_parameter("base_frame").as_string();
            wheel_radius_ = this->get_parameter("wheel_radius").as_double();
            ticks_per_rev_ = this->get_parameter("ticks_per_rev").as_double();
            wheel_base_ = this->get_parameter("wheel_base").as_double();
            do_broadcast_transform_ = this->get_parameter("do_broadcast_transform").as_bool();
            pose_variances_ = this->get_parameter("pose_variances").as_double_array();
            twist_variances_ = this->get_parameter("twist_variances").as_double_array();
            max_lookback_t_ = this->get_parameter("max_history_time").as_double();
            linear_var_ = this->get_parameter("linear_var").as_double();
            angular_var_ = this->get_parameter("angular_var").as_double();

            encoder_subscription_ = this->create_subscription<racing_bot_interfaces::msg::EncoderValues>(
                encoder_topic, encoder_queue_size,
                std::bind(&OdometryNode::EncoderCallback, this, std::placeholders::_1));
            odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(odometry_topic, odometry_queue_size);

            tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        void OdometryNode::EncoderCallback(const racing_bot_interfaces::msg::EncoderValues msg) {
            // The very first message
            if (first_update) {
                prev_msg = msg;
                first_update = false;
                return;
            }

            updatePose(msg);
            calculateSpeed();
            publishOdometry();
            prev_msg = msg;
        }

        // Not used; has problems
        std::vector<float> OdometryNode::CalculatePoseVariances(float d_distance, float d_angle) {
            // angles should be in radians
            float position_var, orientation_var;
            // orientation_var = pow(d_angle, 2) * pow(angular_var_, 2);
        }

        // The robot is a differential-drive robot
        void OdometryNode::updatePose(racing_bot_interfaces::msg::EncoderValues msg) {
            // encoder difference
            auto diff_left =
                (msg.left_encoder.data - prev_msg.left_encoder.data) / ticks_per_rev_ * wheel_radius_ * 2 * M_PI;
            auto diff_right =
                (msg.right_encoder.data - prev_msg.right_encoder.data) / ticks_per_rev_ * wheel_radius_ * 2 * M_PI;
            double dt = (rclcpp::Time(msg.header.stamp) - rclcpp::Time(prev_msg.header.stamp)).seconds();

            // orientation update
            float dtheta = (diff_right - diff_left) / wheel_base_;

            // position update
            float corner_radius, dx, dy;
            if (dtheta != 0) {
                corner_radius = (diff_right + diff_left) / (2 * dtheta);
                dy = corner_radius * sin(dtheta);
                dx = corner_radius * (1 - cos(dtheta));
            } else {
                dy = diff_left;
                dx = 0;
            }

            // update position and account for current orientation
            position_x_ += sin(orientation_) * dx + cos(orientation_) * dy;
            position_y_ += cos(orientation_) * dx + sin(orientation_) * dy;
            orientation_ += dtheta;

            pose_changes.push_front(deltaPose({rclcpp::Time(msg.header.stamp), dt, dx, dy, dtheta}));
        }

        void OdometryNode::calculateSpeed() {
            double distance = 0, angle = 0, time = 0;
            for (auto it = pose_changes.crbegin(); it != pose_changes.crend(); it++) {
                // Remove too old pose changes. Assumes that the values are chronologically ordered.
                if (it->stamp < (this->now() - rclcpp::Duration::from_seconds(max_lookback_t_))) {
                    pose_changes.pop_back();
                } else {
                    distance += sqrt(pow(it->dx, 2) + pow(it->dy, 2));
                    angle += it->dtheta;
                    time += it->dt;
                }
            }
            if (time == 0.) {
                RCLCPP_WARN(this->get_logger(), "The total time for velocity calculations is 0. Queue size: %lu",
                            pose_changes.size());
            } else {
                linear_speed_ = distance / time;
                angular_speed_ = angle / time;
            }
        }

        void OdometryNode::publishOdometry() {
            // Create the odom message
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = this->now();
            odom.header.frame_id = odom_frame_;
            odom.child_frame_id = base_frame_;

            // Set the pose
            geometry_msgs::msg::PoseWithCovariance pose_msg;
            pose_msg.pose.position.x = position_x_;
            pose_msg.pose.position.y = position_y_;
            pose_msg.pose.position.z = 0.0;

            geometry_msgs::msg::Quaternion odometry_quat =
                tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1), orientation_));
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
            twist_msg.twist.linear.x = linear_speed_;
            twist_msg.twist.linear.y = 0.;
            twist_msg.twist.angular.z = angular_speed_;

            twist_msg.covariance[0] = twist_variances_[0];
            twist_msg.covariance[7] = twist_variances_[1];
            twist_msg.covariance[14] = twist_variances_[2];
            twist_msg.covariance[21] = twist_variances_[3];
            twist_msg.covariance[28] = twist_variances_[4];
            twist_msg.covariance[35] = twist_variances_[5];

            odom.twist = twist_msg;

            // Publish the odom message
            odometry_publisher_->publish(odom);

            if (do_broadcast_transform_) {
                RCLCPP_INFO_ONCE(this->get_logger(), "Transform broadcasted by odometry node");

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
        }
    }  // namespace odometry
}  // namespace racing_bot