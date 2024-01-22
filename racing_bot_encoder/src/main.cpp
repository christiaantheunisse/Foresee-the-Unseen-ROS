#include "encoder_node.hpp"

// Encoder pins
const int LEFT_PIN_A = 16, LEFT_PIN_B = 19, RIGHT_PIN_A = 20, RIGHT_PIN_B = 21;

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto encoder_node = std::make_shared<racing_bot::encoder::EncoderNode>(LEFT_PIN_A, LEFT_PIN_B, RIGHT_PIN_A, RIGHT_PIN_B);
  rclcpp::spin(encoder_node);
  rclcpp::shutdown();
  return 0;
}