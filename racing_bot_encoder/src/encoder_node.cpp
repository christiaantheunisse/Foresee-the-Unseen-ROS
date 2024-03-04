#include "encoder_node.hpp"

#include "encoder_constants.hpp"


// Node cannot run without pigpiod ($ sudo pigpiod)
namespace racing_bot {
    namespace encoder {
        EncoderNode::EncoderNode(const int left_pin_a, const int left_pin_b, const int right_pin_a,
                                 const int right_pin_b)
            : rclcpp::Node(NODE_NAME),
              pi_handle_(pigpio_start(NULL, NULL)),
              left_encoder_(pi_handle_, left_pin_a, left_pin_b),
              right_encoder_(pi_handle_, right_pin_a, right_pin_b) {
            
            encoder_publisher_ =
                this->create_publisher<racing_bot_interfaces::msg::EncoderValues>(PUBLISHER_TOPIC, WHEEL_QUEUE_SIZE);
            publish_timer_ = this->create_wall_timer(std::chrono::milliseconds(PUBLISH_RATE),
                                                     std::bind(&EncoderNode::publishMessage, this));
        }

        EncoderNode::~EncoderNode() { pigpio_stop(pi_handle_); }

        void EncoderNode::publishMessage() {
            // std_msgs::msg::Int32 right_count, left_count;

            racing_bot_interfaces::msg::EncoderValues encoder_values;

            encoder_values.header.stamp = this->now();
            // encoder_values.header = std_msgs::msg::Header(this->now());
            std_msgs::msg::Int32 right_reading, left_reading;
            right_reading.data = right_encoder_.getPosition();
            left_reading.data = left_encoder_.getPosition();

            encoder_values.right_encoder = right_reading;
            encoder_values.left_encoder = left_reading;
            encoder_publisher_->publish(encoder_values);

            // left_count.data = left_encoder_.getPosition();
            // right_count.data = right_encoder_.getPosition();
            // left_publisher_->publish(left_count);

            // std_msgs::msg::Int32 right_count;
            // right_publisher_->publish(right_count);
        }
    }  // namespace encoder
}  // namespace racing_bot
