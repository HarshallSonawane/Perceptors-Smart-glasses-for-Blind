#!/usr/bin/python
import RPi.GPIO as GPIO
from time import sleep
import time
import cv2


# def buzz():
def CamAction():
    thres = 0.45  # Threshold to detect object

#  cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)
#cap.set(10, 70)

    classNames = []
    classFile = '/home/pi/ObjectDetectionUsingSSDMobinet-/coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('n').split('n')

    configPath = '/home/pi/ObjectDetectionUsingSSDMobinet-/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = '/home/pi/ObjectDetectionUsingSSDMobinet-/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)


def getObjects(img, draw=True):
    classIds, confs, bbox = net.detect(
        img, confThreshold=thres, nmsThreshold=0.2)
    #print(classIds, bbox)
    objectInfo = []

    with open(classFile, 'rt') as f:
        classNames = [line.rstrip() for line in f]

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId-1]
            objectInfo.append([box, className])
            if (draw):
                cv2.rectangle(img, box, color=(0, 355, 0), thickness=2)
                cv2.putText(img, className.upper(), (box[0]+10, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(round(confidence*100, 2)), (box[0]+200, box[1]+30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                print(classNames[classId-1].upper())
    return img, objectInfo

    # with open(classFile,'rt') as f:
  #  classNames=[line.rstrip() for line in f]


while True:
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
        if distance < 100:
            CamAction
            if __name__ == "__main__":
                cap = cv2.VideoCapture(-1)
                cap.set(3, 640)
                cap.set(4, 480)
            while True:
                success, img = cap.read()
                results, ObjectInfo = getObjects(img, True)
                print(ObjectInfo)
                cv2.imshow("Output", img)
                cv2.waitKey(1)
                # GPIO.setwarnings(False)
                #buzzer = 16
                #GPIO.setup(buzzer, GPIO.OUT)
                #i = 5

               # while i >= 0:
                #GPIO.output(buzzer, GPIO.HIGH)
                # print("Beep")
                # sleep(0.25)  # Delay in seconds 0.25
                #GPIO.output(buzzer, GPIO.LOW)
                #print("No Beep")
                # sleep(0.25)  # sleeep
                #i = i-1

    finally:
        GPIO.cleanup()  # cleaning


# Libraries

# Disable warnings (optional)
