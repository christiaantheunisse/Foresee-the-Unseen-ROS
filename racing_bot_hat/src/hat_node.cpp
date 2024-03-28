#include "hat_node.hpp"
#include "hat_constants.hpp"
#include <pigpiod_if2.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace std::chrono_literals; // to be able to use ms

namespace racing_bot
{
  namespace hat
  {

// define constants for PCA9685 PWM driver
#define MODE1 0x00
#define MODE2 0x01
#define PRESCALE 0xFE
#define LED0_ON_L 0x06
#define LED0_ON_H 0x07
#define LED0_OFF_L 0x08
#define LED0_OFF_H 0x09
#define ALL_LED_ON_L 0xFA
#define ALL_LED_ON_H 0xFB
#define ALL_LED_OFF_L 0xFC
#define ALL_LED_OFF_H 0xFD
#define RESTART 0x80
#define SLEEP 0x10
#define ALLCALL 0x01
#define INVRT 0x10
#define OUTDRV 0x04
#define HAT_ADDR 0x60

    // pin mappings for motor driver inputs
    const int pwmPin[] = {8, 13, 2, 7};
    const int in1Pin[] = {10, 11, 4, 5};
    const int in2Pin[] = {9, 12, 3, 6};

    HatNode::HatNode() : Node("hat_node")
    {
      this->declare_parameter("left_reverse", false);
      this->declare_parameter("right_reverse", false);
      this->declare_parameter("motor_topic", "cmd_motor");

      _direction[0] = this->get_parameter("left_reverse").as_bool() ? -1 : 1;
      _direction[1] = this->get_parameter("right_reverse").as_bool() ? -1 : 1;
      _motor_topic = this->get_parameter("motor_topic").as_string();

      motor_commands_subscription_ = this->create_subscription<std_msgs::msg::Int16MultiArray>(
          SUBSCRIPTION_TOPIC, MOTOR_QUEUE_SIZE,
          std::bind(&HatNode::motorCommandsCallBack, this, std::placeholders::_1));

      pi_handle_ = pigpio_start(NULL, NULL);
      i2cSetup();
      setAllPWM(0, 0);
      configureModeRegisters();
      disableSleepMode();

    }

    void HatNode::configureModeRegisters()
    {
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE2, OUTDRV);
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE1, ALLCALL);
      usleep(5000);
    }

    void HatNode::disableSleepMode()
    {
      int mode1_register_value = i2c_read_byte(pi_handle_, MODE1);
      mode1_register_value = mode1_register_value & ~SLEEP;
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE1, mode1_register_value);
      usleep(5000);
    }

    void HatNode::motorCommandsCallBack(const std_msgs::msg::Int16MultiArray::SharedPtr motor_message)
    {
      for (int i = 0; i < 4; i++)
      {
        motorSetSpeed(i, _direction[i] * motor_message->data[i]);
      }
    }

    void HatNode::turnOffMotors()
    {
      for (int i = 0; i < 4; ++i)
      {
        motorStop(i);
      }
    }

    void HatNode::setPWMFrequency(int frequency)
    {
      double prescale = calculatePrescaleValue(frequency);
      configureModeAndPrescale(prescale);
    }

    double HatNode::calculatePrescaleValue(int frequency) const
    {
      double prescale_value = 25000000.0;
      prescale_value /= 4096.0;
      prescale_value /= frequency;
      prescale_value -= 1.0;
      return floor(prescale_value + 0.5);
    }

    void HatNode::configureModeAndPrescale(double prescale)
    {
      int old_mode = i2c_read_byte(pi_handle_, MODE1);
      int new_mode = (old_mode & 0x7F) | 0x10;
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE1, new_mode);
      i2c_write_byte_data(pi_handle_, i2c_handle_, PRESCALE, static_cast<int>(floor(prescale)));
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE1, old_mode);
      usleep(5000);
      i2c_write_byte_data(pi_handle_, i2c_handle_, MODE1, old_mode | 0x80);
    }

    void HatNode::setPWM(int channel, int on_value, int off_value)
    {
      i2c_write_byte_data(pi_handle_, i2c_handle_, LED0_ON_L + 4 * channel, on_value & 0xFF);
      i2c_write_byte_data(pi_handle_, i2c_handle_, LED0_ON_H + 4 * channel, on_value >> 8);

      i2c_write_byte_data(pi_handle_, i2c_handle_, LED0_OFF_L + 4 * channel, off_value & 0xFF);
      i2c_write_byte_data(pi_handle_, i2c_handle_, LED0_OFF_H + 4 * channel, off_value >> 8);
    }

    void HatNode::setPin(int pin, int pin_value)
    {
      const int full_on = 4096;
      const int full_off = 0;

      if (pin_value == 0)
      {
        setPWM(pin, full_off, full_on);
      }
      else if (pin_value == 1)
      {
        setPWM(pin, full_on, full_off);
      }
    }

    void HatNode::motorStop(int motor_id)
    {
      setPin(in1Pin[motor_id], 0);
      setPin(in2Pin[motor_id], 0);
    }

    void HatNode::motorSetSpeed(int motor_id, int speed)
    {
      if (motor_id < 0 || motor_id > 3)
      {
        std::cout << "Invalid motor ID." << std::endl;
        return;
      }
      if (speed < -255 || speed > 255)
      {
        std::cout << "Invalid speed." << std::endl;
        return;
      }
      if (speed == 0)
      {
        motorStop(motor_id);
      }
      else if (speed > 0)
      {
        setMotorForward(motor_id, speed);
      }
      else
      {
        setMotorReverse(motor_id, -speed);
      }
    }

    void HatNode::setMotorForward(int motor_id, int speed)
    {
      setPin(in2Pin[motor_id], 0);
      setPin(in1Pin[motor_id], 1);
      setPWM(pwmPin[motor_id], 0, speed * 16);
    }

    void HatNode::setMotorReverse(int motor_id, int speed)
    {
      setPin(in1Pin[motor_id], 0);
      setPin(in2Pin[motor_id], 1);
      setPWM(pwmPin[motor_id], 0, speed * 16);
    }

    void HatNode::i2cSetup()
    {
      i2c_handle_ = i2c_open(pi_handle_, 1, HAT_ADDR, 0);
      if (i2c_handle_ == -1)
      {
        std::cout << "Failed to open I2C." << std::endl;
      }
    }

    void HatNode::setAllPWM(int on_value, int off_value)
    {
      i2c_write_byte_data(pi_handle_, i2c_handle_, ALL_LED_ON_L, on_value & 0xFF);
      i2c_write_byte_data(pi_handle_, i2c_handle_, ALL_LED_ON_H, on_value >> 8);
      i2c_write_byte_data(pi_handle_, i2c_handle_, ALL_LED_OFF_L, off_value & 0xFF);
      i2c_write_byte_data(pi_handle_, i2c_handle_, ALL_LED_OFF_H, off_value >> 8);
    }
  }
}