import RPi.GPIO as GPIO
import time
import os


GPIO.setmode(GPIO.BOARD)

TRIG = 16
ECHO = 18
i = 0

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
os.system("espeak 'Calibrating Distance' ")
time.sleep(2)


try:
    while True:
        str = 'python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model'
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        while GPIO.input(ECHO) == 0:
            pulse_start = time.time()

        while GPIO.input(ECHO) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start

        distance = pulse_duration * 17150

        distance = round(distance+1.15, 2)

        if distance <= 150:
            os.system("espeak 'Object Detected at ' "+distance+" ' metres ' ")
            time.sleep(2)
            os.system("espeak 'Launching Camera for Object Detection' ")
            time.sleep(2)
            os.system(str)


except KeyboardInterrupt:
    GPIO.cleanup()
