#include <gtest/gtest.h>
#include "hat_node.hpp"

TEST(HatNodeTest, ConstructorTest)
{
  rclcpp::init(0, nullptr);
  auto hat_node = std::make_shared<racing_bot::hat::HatNode>();

  EXPECT_TRUE(hat_node != nullptr);

  rclcpp::shutdown();
}

TEST(HatNodeTest, TurnOffMotorsTest)
{
  rclcpp::init(0, nullptr);
  auto hat_node = std::make_shared<racing_bot::hat::HatNode>();
  EXPECT_NO_THROW(hat_node->turnOffMotors());
  rclcpp::shutdown();
}

TEST(HatNodeTest, SetPWMFrequencyTest)
{
  const int FREQUENCY = 100;
  rclcpp::init(0, nullptr);
  auto hat_node = std::make_shared<racing_bot::hat::HatNode>();
  EXPECT_NO_THROW(hat_node->setPWMFrequency(FREQUENCY));
  rclcpp::shutdown();
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
