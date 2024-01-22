#include <gtest/gtest.h>
#include "encoder_node.hpp"
#include "encoder_sensor.hpp"

TEST(EncoderNodeTest, ConstructorTest)
{
  const int LEFT_PIN_A = 16, LEFT_PIN_B = 19, RIGHT_PIN_A = 20, RIGHT_PIN_B = 21;

  rclcpp::init(0, nullptr);
  auto encoder_node = std::make_shared<racing_bot::encoder::EncoderNode>(LEFT_PIN_A, LEFT_PIN_B, RIGHT_PIN_A, RIGHT_PIN_B);
  EXPECT_TRUE(encoder_node != nullptr);
  rclcpp::shutdown();
}

TEST(EncoderSensorTest, GetPositionTest)
{
  const int PI_HANDLE = 0;
  const unsigned int PIN_A = 1, PIN_B = 2;

  racing_bot::encoder::EncoderSensor encoder_sensor(PI_HANDLE, PIN_A, PIN_B);
  int position = encoder_sensor.getPosition();
  EXPECT_EQ(position, 0);
}

int main(int argc, char **argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
