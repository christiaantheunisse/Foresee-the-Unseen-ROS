#include "imu_node.hpp"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "imu_constants.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_sensor_msgs.h"

namespace racing_bot {
    namespace imu {

#define LSM9DS1_AG_ADDRESS 0x6b  // Accelerometer & Gyro Address
#define LSM9DS1_M_ADDRESS 0x1e   // Magnetometer Address

        ImuNode::ImuNode() : Node(NODE_NAME), filter_(), imu_(IMU_MODE_I2C, 0x6b, 0x1e) {
            imu_values_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(PUBLISHER_TOPIC, IMU_QUEUE_SIZE);
            publish_timer_ = this->create_wall_timer(std::chrono::milliseconds(TIMER_MILLISECONDS),
                                                     std::bind(&ImuNode::readImuValues, this));
        }

        void ImuNode::startDevice() {
            imu_.begin();
            if (!imu_.begin()) {
                fprintf(stderr, "Failed to communicate with LSM9DS1.\n");
                exit(EXIT_FAILURE);
            }
            imu_.calibrate();
            filter_.begin(FILTER_SAMPLE_FREQUENCY);
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

            // filter_.updateIMU(gyroscope_x, gyroscope_y, gyroscope_z, accelerometer_x, accelerometer_y,
            // accelerometer_z); filter_.update(gyroscope_x, gyroscope_y, gyroscope_z, accelerometer_x, accelerometer_y,
            // accelerometer_z, magnetometer_x, magnetometer_y, magnetometer_z);

            // Convert LHS to RHS: [x, y, z] --> [y, x, z]
            filter_.update(gyroscope_y, gyroscope_x, gyroscope_z, accelerometer_y, accelerometer_x, accelerometer_z,
                           magnetometer_x, magnetometer_y, magnetometer_z);

            // Convert LHS to RHS: [x, y, z] --> [y, x, z]
            publishImuMessage(accelerometer_y, accelerometer_x, accelerometer_z, gyroscope_y, gyroscope_x, gyroscope_z);
        }

        void ImuNode::publishImuMessage(const float accelerometer_x, const float accelerometer_y,
                                        const float accelerometer_z, const float gyroscope_x, const float gyroscope_y,
                                        const float gyroscope_z) {
            sensor_msgs::msg::Imu imu_data;

            imu_data.header.stamp = this->now();
            imu_data.header.frame_id = FRAME_ID;

            imu_data.linear_acceleration.x = accelerometer_x;
            imu_data.linear_acceleration.y = accelerometer_y;
            imu_data.linear_acceleration.z = accelerometer_z;

            imu_data.angular_velocity.x = gyroscope_x;
            imu_data.angular_velocity.y = gyroscope_y;
            imu_data.angular_velocity.z = gyroscope_z;

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
            std::string goal_frame = "base_link";
            try {
                t = tf_buffer_->lookupTransform(goal_frame, FRAME_ID, tf2::TimePointZero);
            } catch (const tf2::TransformException& ex) {
                RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s", FRAME_ID.c_str(),
                            goal_frame.c_str(), ex.what());
                return;
            }
            tf2::doTransform(imu_data, imu_data_transformed, t)

            imu_values_publisher_->publish(imu_data_transformed);
        }
    }  // namespace imu
}  // namespace racing_bot