#include "odometry_node.hpp"
#include "odometry_constants.hpp"

const double WHEEL_RADIUS = 0.033;
const double TICKS_PER_REV = 1920.0;
const double WHEEL_BASE = 0.153;

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto odom_node = std::make_shared<racing_bot::odometry::OdometryNode>(WHEEL_RADIUS, TICKS_PER_REV, WHEEL_BASE);
  rclcpp::spin(odom_node);
  rclcpp::shutdown();
  return 0;
}