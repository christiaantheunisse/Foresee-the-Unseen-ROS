#include "encoder_node.hpp"

#include "encoder_constants.hpp"

// Node cannot run without pigpiod ($ sudo pigpiod)
namespace racing_bot {
    namespace encoder {
        EncoderNode::EncoderNode()
            : rclcpp::Node("encoder_node"),
              pi_handle_(pigpio_start(NULL, NULL))
            {
            this->declare_parameter("encoder_topic", "wheel_encoders");
            this->declare_parameter("update_frequency", 50.);  // [Hz]
            this->declare_parameter("left_pin_a", 16); 
            this->declare_parameter("left_pin_b", 19); 
            this->declare_parameter("right_pin_a", 20);
            this->declare_parameter("right_pin_b", 21);


            encoder_topic_ = this->get_parameter("encoder_topic").as_string();
            frequency_ = this->get_parameter("update_frequency").as_double();
            int left_pin_a = this->get_parameter("left_pin_a").as_int();
            int left_pin_b = this->get_parameter("left_pin_b").as_int();
            int right_pin_a = this->get_parameter("right_pin_a").as_int();
            int right_pin_b = this->get_parameter("right_pin_b").as_int();


            left_encoder_ = std::make_shared<EncoderSensor>(pi_handle_, left_pin_a, left_pin_b);
            right_encoder_ = std::make_shared<EncoderSensor>(pi_handle_, right_pin_a, right_pin_b);
            

            encoder_publisher_ =
                this->create_publisher<racing_bot_interfaces::msg::EncoderValues>(encoder_topic_, 5);
            publish_timer_ = this->create_wall_timer(std::chrono::milliseconds((int) (1 / frequency_ * 1000)),
                                                     std::bind(&EncoderNode::publishMessage, this));
        }

        EncoderNode::~EncoderNode() { pigpio_stop(pi_handle_); }

        void EncoderNode::publishMessage() {

            racing_bot_interfaces::msg::EncoderValues encoder_values;

            encoder_values.header.stamp = this->now();

            // std_msgs::msg::Int32 right_reading, left_reading;
            // right_reading.data = right_encoder_.getPosition();
            // left_reading.data = left_encoder_.getPosition();

            encoder_values.right_encoder.data = right_encoder_->getPosition();
            encoder_values.left_encoder.data = left_encoder_->getPosition();
            encoder_publisher_->publish(encoder_values);

        }
    }  // namespace encoder
}  // namespace racing_bot
