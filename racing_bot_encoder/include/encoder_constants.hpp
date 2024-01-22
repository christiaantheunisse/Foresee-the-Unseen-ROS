#ifndef ENCODERCONSTANTS_H
#define ENCODERCONSTANTS_H

#include <string>

namespace racing_bot
{
  namespace encoder
  {
    const int WHEEL_QUEUE_SIZE = 1;
    const std::string LEFT_PUBLISHER_TOPIC = "left_wheel";
    const std::string RIGHT_PUBLISHER_TOPIC = "right_wheel";
    const int PUBLISH_RATE = 33;
    const std::string NODE_NAME = "encoder_node";
  }
}

#endif