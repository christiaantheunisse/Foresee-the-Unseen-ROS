#include <gtest/gtest.h>
#include "odometry_node.hpp"

TEST(OdometryNodeTest, ConstructorTest)
{
  const double wheel_radius = 1.0;
  const double ticks_per_rev = 1.0;
  const double wheel_base = 1.0;

  rclcpp::init(0, nullptr);
  auto odom_node = std::make_shared<racing_bot::odometry::OdometryNode>(
      wheel_radius, ticks_per_rev, wheel_base);

  EXPECT_TRUE(odom_node != nullptr);

  rclcpp::shutdown();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
