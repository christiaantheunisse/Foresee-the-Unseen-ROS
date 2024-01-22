#ifndef ODOMCONSTANTS_H
#define ODOMCONSTANTS_H

#include <string>

namespace racing_bot
{
  namespace odometry
  {

    const std::string NODE_NAME = "odom_node";
    const std::string PUBLISHER_TOPIC = "odom";
    const std::string FRAME_ID = "odom";
    const std::string CHILD_FRAME_ID = "base_link";
    const int ODOMETRY_QUEUE_SIZE = 5;
    const int ENCODER_QUEUE_SIZE = 1;
    const std::string LEFT_SUBSCRIBER_TOPIC = "left_wheel";
    const std::string RIGHT_SUBSCRIBER_TOPIC = "right_wheel";
  }
}
#endif
