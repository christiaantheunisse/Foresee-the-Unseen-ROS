#include "imu_node.hpp"

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  auto imu_node = std::make_shared<racing_bot::imu::ImuNode>();
  imu_node->startDevice();
  rclcpp::spin(imu_node);
  rclcpp::shutdown();
  return 0;
}