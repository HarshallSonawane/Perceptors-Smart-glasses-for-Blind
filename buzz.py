#!/usr/bin/python
import RPi.GPIO as GPIO
import time


try:
    GPIO.setmode(GPIO.BOARD)

    PIN_TRIGGER = 7
    PIN_ECHO = 11

    GPIO.setup(PIN_TRIGGER, GPIO.OUT)
    GPIO.setup(PIN_ECHO, GPIO.IN)

    GPIO.output(PIN_TRIGGER, GPIO.LOW)

    print("Waiting for sensor to settle")

    time.sleep(2)

    print("Calculating distance")

    GPIO.output(PIN_TRIGGER, GPIO.HIGH)

    time.sleep(0.00001)

    GPIO.output(PIN_TRIGGER, GPIO.LOW)

    while GPIO.input(PIN_ECHO) == 0:
        pulse_start_time = time.time()
    while GPIO.input(PIN_ECHO) == 1:
        pulse_end_time = time.time()

    pulse_duration = pulse_end_time - pulse_start_time
    distance = round(pulse_duration * 17150, 2)
    print("Distance:", distance, "cm")
    if distance > 200:
        buzz()


finally:
    GPIO.cleanup()


# Libraries

# Disable warnings (optional)
def buzz():
    GPIO.setwarnings(False)


# Select GPIO mode
GPIO.setmode(GPIO.BCM)
# Set buzzer - pin 23 as output
buzzer = 23
GPIO.setup(buzzer, GPIO.OUT)
# Run forever loop
while True:
    GPIO.output(buzzer, GPIO.HIGH)
    print("Beep")
    sleep(0.5)  # Delay in seconds
    GPIO.output(buzzer, GPIO.LOW)
    print("No Beep")
    sleep(0.5)

my_function()