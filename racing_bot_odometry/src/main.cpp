#include "odometry_node.hpp"


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto odom_node = std::make_shared<racing_bot::odometry::OdometryNode>();
  rclcpp::spin(odom_node);
  rclcpp::shutdown();
  return 0;
}