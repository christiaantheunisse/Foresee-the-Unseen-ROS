#ifndef IMUNODE_H
#define IMUNODE_H

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "transform_imu.h"
#include "LSM9DS1_Types.h"
#include "LSM9DS1.h"
#include "MadgwickAHRS.h"

namespace racing_bot
{
  namespace imu
  {
    /**
     * @brief Constructs an IMU Node object.
     *
     * This class represents an IMU Node that reads IMU values and publishes them as sensor_msgs::msg::Imu.
     * The constructor initializes the IMU Node and sets up the publisher and timer for publishing IMU values.
     */
    class ImuNode : public rclcpp::Node
    {
    public:
      ImuNode();
      void startDevice();

    private:
      rclcpp::TimerBase::SharedPtr publish_timer_;
      rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_values_publisher_;
      Madgwick filter_;
      LSM9DS1 imu_;
      void readImuValues();
      void publishImuMessage(const float accelerometer_x, const float accelerometer_y, const float accelerometer_z, const float gyroscope_x, const float gyroscope_y, const float gyroscope_z);

      std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
      std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};

      std::string imu_frame_;
      std::string base_frame_;
      std::string imu_topic_;
      float frequency_;
      bool use_magnetometer_;
      std::vector<double> lin_acc_variances_;
      std::vector<double> ang_vel_variances_;
      std::vector<double> orient_variances_;
      float madgwick_gain_;
      // float freq_scale_factor_; TODO: remove this completely
      float ang_vel_calibrate_factor_;
    };
  }
}
#endif