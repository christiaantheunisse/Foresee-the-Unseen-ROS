#ifndef HATNODE_H
#define HATNODE_H

#include "std_msgs/msg/int16_multi_array.hpp"
#include "rclcpp/rclcpp.hpp"
#include <string>

namespace racing_bot
{
  namespace hat
  {

    /**
     * @brief Represents a HatNode object that controls motors using a motor hat.
     *
     * The HatNode class is responsible for controlling motors using a motor hat.
     * It provides functions to turn off the motors, set the PWM frequency, and handle
     * motor commands received from a the cmd_motor topic. The class communicates with the
     * motor hat through the pigpiod_if2 library / the I2C interface.
     */
    class HatNode : public rclcpp::Node
    {
    public:
      HatNode();
      void turnOffMotors();
      void setPWMFrequency(int frequency);

    private:
      void motorCommandsCallBack(const std_msgs::msg::Int16MultiArray::SharedPtr motor_message);
      void configureModeRegisters();
      void disableSleepMode();
      double calculatePrescaleValue(int frequency) const;
      void configureModeAndPrescale(double prescale);
      void motorSetSpeed(int motor_id, int speed);
      void setPWM(int channel, int on_value, int off_value);
      void setPin(int pin, int pin_value);
      void motorStop(int motor_id);
      void setMotorForward(int motor_id, int speed);
      void setMotorReverse(int motor_id, int speed);
      void i2cSetup();
      void setAllPWM(int on_value, int off_value);

      rclcpp::Subscription<std_msgs::msg::Int16MultiArray>::SharedPtr motor_commands_subscription_;
      int pi_handle_;
      int i2c_handle_;

      int _direction[4];
      std::string _motor_topic;
    };
  }
}

#endif
