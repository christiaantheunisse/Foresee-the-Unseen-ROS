#ifndef IMUNODE_H
#define IMUNODE_H

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
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
    };
  }
}
#endif