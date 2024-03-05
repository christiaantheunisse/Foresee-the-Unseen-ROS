#include <string>

#ifndef ENCODERCONSTANTS_H
#define ENCODERCONSTANTS_H

namespace racing_bot {
    namespace encoder {
        const int WHEEL_QUEUE_SIZE = 1;
        // const std::string LEFT_PUBLISHER_TOPIC = "left_wheel";
        // const std::string RIGHT_PUBLISHER_TOPIC = "right_wheel";
        const std::string PUBLISHER_TOPIC = "wheel_encoders";
        const int PUBLISH_RATE = 33;
        const std::string NODE_NAME = "encoder_node";
    }  // namespace encoder
}  // namespace racing_bot

#endif