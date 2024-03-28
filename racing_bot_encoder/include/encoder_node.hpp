#include <pigpiod_if2.h>

#include <memory>
#include <string>

#include "encoder_sensor.hpp"
#include "racing_bot_interfaces/msg/encoder_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"
// #include "racing_bot_interfaces/msg/trajectory.hpp"

#ifndef ENCODERNODE_H
#define ENCODERNODE_H
namespace racing_bot {
    namespace encoder {
        /**
         * @brief Constructs an Encoder Node object with the specified pin configurations.
         *
         * This class represents an Encoder Node that reads encoder positions and publishes them as
         * std_msgs::msg::Int32. The constructor initializes the Encoder Node with the given pin configurations for the
         * left and right encoders. It sets up the publishers and timer for periodically publishing the encoder
         * positions.
         *
         * @param left_pin_a The GPIO pin connected to pin A of the left encoder.
         * @param left_pin_b The GPIO pin connected to pin B of the left encoder.
         * @param right_pin_a The GPIO pin connected to pin A of the right encoder.
         * @param right_pin_b The GPIO pin connected to pin B of the right encoder.
         */
        class EncoderNode : public rclcpp::Node {
           public:
            EncoderNode();
            ~EncoderNode();

           private:
            void publishMessage();
            rclcpp::Publisher<racing_bot_interfaces::msg::EncoderValues>::SharedPtr encoder_publisher_;
            rclcpp::TimerBase::SharedPtr publish_timer_;
            const int pi_handle_;
            std::shared_ptr<EncoderSensor> left_encoder_;
            std::shared_ptr<EncoderSensor> right_encoder_;

            std::string encoder_topic_;
            float frequency_;
        };
    }  // namespace encoder
}  // namespace racing_bot
#endif