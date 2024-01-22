#include "controller_node.hpp"

const double ROBOT_WIDTH = 0.15;

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto controllerNode = std::make_shared<racing_bot::controller::ControllerNode>(ROBOT_WIDTH);
  rclcpp::spin(controllerNode);
  rclcpp::shutdown();

  return 0;
}
