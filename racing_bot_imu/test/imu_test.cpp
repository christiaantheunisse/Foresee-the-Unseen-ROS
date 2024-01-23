#include <gtest/gtest.h>
#include "imu_node.hpp"

TEST(ImuNodeTest, ConstructorTest)
{
  rclcpp::init(0, nullptr);
  auto imu_node = std::make_shared<racing_bot::imu::ImuNode>();
  EXPECT_TRUE(imu_node != nullptr);
  rclcpp::shutdown();
}

TEST(ImuNodeTest, StartDeviceTest)
{
  rclcpp::init(0, nullptr);
  auto imu_node = std::make_shared<racing_bot::imu::ImuNode>();
  EXPECT_NO_THROW(imu_node->startDevice());
  rclcpp::shutdown();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
