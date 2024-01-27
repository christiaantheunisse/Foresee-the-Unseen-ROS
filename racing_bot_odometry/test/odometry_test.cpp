#include <gtest/gtest.h>
#include "odometry_node.hpp"

TEST(OdometryNodeTest, ConstructorTest)
{
  rclcpp::init(0, nullptr);
  auto odom_node = std::make_shared<racing_bot::odometry::OdometryNode>();

  EXPECT_TRUE(odom_node != nullptr);

  rclcpp::shutdown();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
