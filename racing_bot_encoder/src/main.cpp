#include "encoder_node.hpp"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto encoder_node = std::make_shared<racing_bot::encoder::EncoderNode>();
  rclcpp::spin(encoder_node);
  rclcpp::shutdown();
  return 0;
}