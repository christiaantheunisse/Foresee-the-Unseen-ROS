#include "hat_node.hpp"

const int FREQUENCY = 1600;

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto hat_node = std::make_shared<racing_bot::hat::HatNode>();
  hat_node->setPWMFrequency(FREQUENCY);
  rclcpp::spin(hat_node);
  rclcpp::shutdown();
  hat_node->turnOffMotors();

  return 0;
}