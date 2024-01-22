#include "controller_node.hpp"
#include "controller_constants.hpp"

namespace racing_bot
{
  namespace controller
  {

    ControllerNode::ControllerNode(double robot_width) : Node(NODE_NAME), robot_width_(robot_width)
    {
      twist_subscription_ = create_subscription<geometry_msgs::msg::Twist>(SUBSCRIBER_TOPIC, VELOCITY_QUEUE_SIZE, std::bind(&ControllerNode::twistCallBack, this, std::placeholders::_1));
      motor_commands_publisher_ = create_publisher<std_msgs::msg::Int16MultiArray>(PUBLISHER_TOPIC, MOTOR_QUEUE_SIZE);
    }

    void ControllerNode::twistCallBack(const geometry_msgs::msg::Twist::SharedPtr twist_message)
    {
      double left_motor_speed, right_motor_speed;
      convertToMotorSpeeds(left_motor_speed, right_motor_speed, twist_message->linear.x, twist_message->angular.z);
      convertToPwm(left_motor_speed, right_motor_speed);
      publishPwm(left_motor_speed, right_motor_speed);
    }

    void ControllerNode::convertToMotorSpeeds(double &left_motor_speed, double &right_motor_speed, const double linear_velocity, double angular_velocity) const
    {
      angular_velocity *= -1;
      const double linear_direction = getDirection(linear_velocity);
      const double angular_direction = getDirection(angular_velocity);
      computeMotorSpeeds(left_motor_speed, right_motor_speed, linear_velocity, angular_velocity, linear_direction, angular_direction);
    }

    int ControllerNode::getDirection(const double velocity) const
    {
      return (velocity < 0) ? -1 : 1;
    }

    void ControllerNode::computeMotorSpeeds(double &left_motor_speed, double &right_motor_speed, const double linear_velocity, const double angular_velocity, const double linear_direction, const double angular_direction) const
    {
      double absolute_linear_velocity = fabs(linear_velocity);
      double absolute_angular_velocity = fabs(angular_velocity);

      double angular_delta = (robot_width_ / 2) * absolute_angular_velocity;

      if (angular_direction > 0)
      {
        left_motor_speed = (absolute_linear_velocity + angular_delta) * linear_direction;
        right_motor_speed = (absolute_linear_velocity - angular_delta) * linear_direction;
      }
      else
      {
        left_motor_speed = (absolute_linear_velocity - angular_delta) * linear_direction;
        right_motor_speed = (absolute_linear_velocity + angular_delta) * linear_direction;
      }
    }

    void ControllerNode::convertToPwm(double &left_motor_speed, double &right_motor_speed) const
    {
      scaleMotorSpeeds(left_motor_speed, right_motor_speed);
      left_motor_speed *= 255;
      right_motor_speed *= 255;
    }

    void ControllerNode::scaleMotorSpeeds(double &left_motor_speed, double &right_motor_speed) const
    {
      double scale_factor;

      if (left_motor_speed > 1 || right_motor_speed > 1)
      {
        scale_factor = 1 / std::max(left_motor_speed, right_motor_speed);
        left_motor_speed *= scale_factor;
        right_motor_speed *= scale_factor;
      }
      else if (left_motor_speed < -1 || right_motor_speed < -1)
      {
        scale_factor = -1 / std::min(left_motor_speed, right_motor_speed);
        left_motor_speed *= scale_factor;
        right_motor_speed *= scale_factor;
      }
    }

    void ControllerNode::publishPwm(const double &left_motor_speed, const double &right_motor_speed)
    {
      std_msgs::msg::Int16MultiArray motor_message;
      motor_message.data.push_back((int)left_motor_speed);
      motor_message.data.push_back((int)right_motor_speed);
      motor_message.data.push_back(0);
      motor_message.data.push_back(0);
      motor_commands_publisher_->publish(motor_message);
    }
  }
}