#ifndef CONTROLLERCONSTANTS_H
#define CONTROLLERCONSTANTS_H

#include <string>

namespace racing_bot
{
  namespace controller
  {

    const int VELOCITY_QUEUE_SIZE = 1;
    const int MOTOR_QUEUE_SIZE = 5;
    const std::string NODE_NAME = "controller_node";
    const std::string PUBLISHER_TOPIC = "cmd_motor";
    const std::string SUBSCRIBER_TOPIC = "cmd_vel";
  }
}

#endif