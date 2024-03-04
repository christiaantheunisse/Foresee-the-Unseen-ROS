#include <stdint.h>

#ifndef ENCODERSENSOR_H
#define ENCODERSENSOR_H

namespace racing_bot {
    namespace encoder {
        /**
         * @brief Constructs an EncoderSensor object with the specified parameters.
         *
         * This constructor initializes an EncoderSensor object with the given Pi handle, pin A, and pin B.
         * It sets up the encoder sensor with the specified parameters for reading the position.
         *
         * @param pi_handle The handle to the pigpio library.
         * @param pin_a The GPIO pin connected to pin A of the encoder sensor.
         * @param pin_b The GPIO pin connected to pin B of the encoder sensor.
         */
        class EncoderSensor {
           public:
            EncoderSensor(const int pi_handle, const unsigned int pin_a, const unsigned int pin_b);
            /**
             * @brief Get the current position of the encoder sensor.
             *
             * This function returns the current position of the specific encoder sensor.
             *
             * @return The current position of the encoder sensor.
             */
            int getPosition() const;

           private:
            const int pi_handle_;
            const unsigned int pin_a_;
            const unsigned int pin_b_;
            int position_;
            int state_;
            static void updatePosition(int pi_handle, unsigned int gpio, unsigned int level, uint32_t tick,
                                       void *userdata);
            void comparePreviousStates(const int state);
        };
    }  // namespace encoder
}  // namespace racing_bot
#endif