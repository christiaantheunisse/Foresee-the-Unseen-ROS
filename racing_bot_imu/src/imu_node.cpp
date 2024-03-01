#include "imu_node.hpp"

namespace racing_bot {
    namespace imu {

#define LSM9DS1_AG_ADDRESS 0x6b  // Accelerometer & Gyro Address
#define LSM9DS1_M_ADDRESS 0x1e   // Magnetometer Address

        ImuNode::ImuNode() : Node("imu_node"), filter_(), imu_(IMU_MODE_I2C, 0x6b, 0x1e) {
            this->declare_parameter("imu_frame", "imu_link");
            this->declare_parameter("base_frame", "base_link");
            this->declare_parameter("imu_topic", "imu_data");
            this->declare_parameter("update_frequency", 100.);  // [Hz]
            this->declare_parameter("use_magnetometer", true);

            imu_frame_ = this->get_parameter("imu_frame").as_string();
            base_frame_ = this->get_parameter("base_frame").as_string();
            imu_topic_ = this->get_parameter("imu_topic").as_string();
            frequency_ = this->get_parameter("update_frequency").as_double();
            use_magnetometer_ = this->get_parameter("use_magnetometer").as_bool();

            // convert the frequency to a time interval in integer milliseconds and update the frequency
            int time_interval = (int)1 / frequency_ * 1000;  // [ms]
            frequency_ = 1000. / time_interval;

            imu_values_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(imu_topic_, 20);
            publish_timer_ = this->create_wall_timer(std::chrono::milliseconds(time_interval),
                                                     std::bind(&ImuNode::readImuValues, this));
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        }

        void ImuNode::startDevice() {
            imu_.begin();
            if (!imu_.begin()) {
                fprintf(stderr, "Failed to communicate with LSM9DS1.\n");
                exit(EXIT_FAILURE);
            }
            imu_.calibrate();
            filter_.begin(frequency_);
        }

        void ImuNode::readImuValues() {
            while (!imu_.gyroAvailable())
                ;
            imu_.readGyro();
            while (!imu_.accelAvailable())
                ;
            imu_.readAccel();
            while (!imu_.magAvailable())
                ;
            imu_.readMag();

            // The board has all three sensors:
            // Datasheet: https://www.st.com/resource/en/datasheet/lsm9ds1.pdf
            // FS = full scale

            // Unit is g (gravitational constant), but should be m/s^2 for ROS msg; so multiply by g=9.81
            // left-handed coordinate system
            float g = 9.81;
            float accelerometer_x = imu_.calcAccel(imu_.ax) * g;
            float accelerometer_y = imu_.calcAccel(imu_.ay) * g;
            float accelerometer_z = imu_.calcAccel(imu_.az) * g;

            // Unit is dps (degrees/sec), but should be rad/s for ROS msg and Madgwick;
            // left-handed coordinate system
            float deg_to_rad = 0.0174533;  // pi / 180
            float gyroscope_x = imu_.calcGyro(imu_.gx) * deg_to_rad;
            float gyroscope_y = imu_.calcGyro(imu_.gy) * deg_to_rad;
            float gyroscope_z = imu_.calcGyro(imu_.gz) * deg_to_rad;

            // Unit is Gauss
            // right-handed coordinate system
            float magnetometer_x = imu_.calcMag(imu_.mx);
            float magnetometer_y = imu_.calcMag(imu_.my);
            float magnetometer_z = imu_.calcMag(imu_.mz);

            // Madgwick algorithm normalizes the accelerometer and magnetometer measurements
            //  and takes the gyroscope measurements in rad/sec

            // Convert LHS to RHS: [x, y, z] --> [-y, -x, z]
            if ((magnetometer_x == 0.0f) && (magnetometer_y == 0.0f) && (magnetometer_z == 0.0f)) {
                RCLCPP_WARN_ONCE(this->get_logger(), "No magnetometer readings available");
            }
            if (use_magnetometer_) {
                filter_.update(gyroscope_y, gyroscope_x, gyroscope_z, accelerometer_y, accelerometer_x, accelerometer_z,
                               magnetometer_x, magnetometer_y, magnetometer_z);
            } else {
                filter_.updateIMU(gyroscope_y, gyroscope_x, gyroscope_z, accelerometer_y, accelerometer_x,
                                  accelerometer_z);
            }
            publishImuMessage(accelerometer_y, accelerometer_x, accelerometer_z, gyroscope_y, gyroscope_x, gyroscope_z);
        }

        void ImuNode::publishImuMessage(const float accelerometer_x, const float accelerometer_y,
                                        const float accelerometer_z, const float gyroscope_x, const float gyroscope_y,
                                        const float gyroscope_z) {
            sensor_msgs::msg::Imu imu_data;

            imu_data.header.stamp = this->now();
            imu_data.header.frame_id = imu_frame_;

            imu_data.linear_acceleration.x = accelerometer_x;
            imu_data.linear_acceleration.y = accelerometer_y;
            imu_data.linear_acceleration.z = accelerometer_z;

            imu_data.angular_velocity.x = gyroscope_x;
            imu_data.angular_velocity.y = gyroscope_y;
            imu_data.angular_velocity.z = gyroscope_z;

            // This is the orientation with respect to its initial orientation -- has an initial orientation of pi/2 
            // around the z-axis with respect to the base frame.
            float quaternion[4];
            filter_.getQuaternion(quaternion);
            imu_data.orientation.w = quaternion[0];
            imu_data.orientation.x = quaternion[1];
            imu_data.orientation.y = quaternion[2];
            imu_data.orientation.z = quaternion[3];

            imu_data.linear_acceleration_covariance[0] = 0.5;
            imu_data.linear_acceleration_covariance[4] = 0.5;
            imu_data.linear_acceleration_covariance[8] = 0.5;

            imu_data.angular_velocity_covariance[0] = 0.1;
            imu_data.angular_velocity_covariance[4] = 0.1;
            imu_data.angular_velocity_covariance[8] = 0.1;

            // Transfer the message to the robot base_link frame
            sensor_msgs::msg::Imu imu_data_transformed;
            geometry_msgs::msg::TransformStamped t;
            try {
                t = tf_buffer_->lookupTransform(base_frame_, imu_frame_, tf2::TimePointZero);
            } catch (const tf2::TransformException& ex) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Could not transform %s to %s: %s",
                                     imu_frame_.c_str(), base_frame_.c_str(), ex.what());
                return;
            }
            tf2::doTransform(imu_data, imu_data_transformed, t);

            imu_values_publisher_->publish(imu_data_transformed);
        }
    }  // namespace imu
}  // namespace racing_bot