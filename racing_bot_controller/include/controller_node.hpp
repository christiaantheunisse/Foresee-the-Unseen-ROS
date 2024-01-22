#ifndef CONTROLLERNODE_H
#define CONTROLLERNODE_H

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int16_multi_array.hpp"

namespace racing_bot
{
  namespace controller
  {

    /**
     * @brief Constructs a ControllerNode object with the specified robot width.
     *
     * This constructor initializes a ControllerNode object with the given robot width.
     * It subscribes to the "cmd_vel" topic to receive Twist messages, converts the linear
     * and angular velocities into motor speeds, and then converts the motor speeds to PWM
     * values readable by the motor hat. Finally, it publishes the motor commands on the
     * "cmd_motor" topic.
     *
     * @param robot_width The width of the robot.
     */
    class ControllerNode : public rclcpp::Node
    {
    public:
      ControllerNode(const double robot_width);

    private:
      rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_subscription_;
      rclcpp::Publisher<std_msgs::msg::Int16MultiArray>::SharedPtr motor_commands_publisher_;
      const double robot_width_;

      void twistCallBack(const geometry_msgs::msg::Twist::SharedPtr twist_message);
      void convertToMotorSpeeds(double &left_motor_speed, double &right_motor_speed, const double linear_velocity, double angular_velocity) const;
      void convertToPwm(double &left_motor_speed, double &right_motor_speed) const;
      int getDirection(double velocity) const;
      void computeMotorSpeeds(double &left_motor_speed, double &right_motor_speed, const double linear_velocity, const double angular_velocity, double linear_direction, double angular_direction) const;
      void scaleMotorSpeeds(double &left_motor_speed, double &right_motor_speed) const;
      void publishPwm(const double &left_motor_speed, const double &right_motor_speed);
    };
  }
}
#endif