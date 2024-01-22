#include <pigpiod_if2.h>
#include "encoder_node.hpp"
#include "encoder_constants.hpp"

namespace racing_bot
{
  namespace encoder
  {
    EncoderNode::EncoderNode(const int left_pin_a, const int left_pin_b, const int right_pin_a, const int right_pin_b)
        : rclcpp::Node(NODE_NAME),
          pi_handle_(pigpio_start(NULL, NULL)),
          left_encoder_(pi_handle_, left_pin_a, left_pin_b),
          right_encoder_(pi_handle_, right_pin_a, right_pin_b)
    {
      left_publisher_ = this->create_publisher<std_msgs::msg::Int32>(LEFT_PUBLISHER_TOPIC, WHEEL_QUEUE_SIZE);
      right_publisher_ = this->create_publisher<std_msgs::msg::Int32>(RIGHT_PUBLISHER_TOPIC, WHEEL_QUEUE_SIZE);
      publish_timer_ = this->create_wall_timer(std::chrono::milliseconds(PUBLISH_RATE), std::bind(&EncoderNode::publishMessage, this));
    }

    EncoderNode::~EncoderNode()
    {
      pigpio_stop(pi_handle_);
    }

    void EncoderNode::publishMessage()
    {
      std_msgs::msg::Int32 left_count;
      left_count.data = left_encoder_.getPosition();
      left_publisher_->publish(left_count);

      std_msgs::msg::Int32 right_count;
      right_count.data = right_encoder_.getPosition();
      right_publisher_->publish(right_count);
    }
  }
}
