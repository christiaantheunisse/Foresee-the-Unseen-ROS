#include <gtest/gtest.h>
#include "controller_node.hpp"

TEST(ControllerNodeTest, ConstructorTest)
{
  const double ROBOT_WIDTH = 0.15;

  rclcpp::init(0, nullptr);
  auto controller_node = std::make_shared<racing_bot::controller::ControllerNode>(ROBOT_WIDTH);

  EXPECT_TRUE(controller_node != nullptr);

  rclcpp::shutdown();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
