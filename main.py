import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import RPi.GPIO as GPIO
import time
import os


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(440, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 520)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC,
                              cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):

        Thread(target=self.update, args=()).start()
        return self

    def update(self):

        while True:

            if self.stopped:

                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):

        return self.frame

    def stop(self):

        self.stopped = True


parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate


if use_TPU:

    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'


CWD_PATH = os.getcwd()


PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)


PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)


with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]


if labels[0] == '???':
    del (labels[0])


if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2


frame_rate_calc = 1
freq = cv2.getTickFrequency()


videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
# time.sleep(1)

distance = 99999999


def buzz():
    GPIO.setmode(GPIO.BOARD)

    TRIG = 16
    ECHO = 18
    i = 0

    GPIO.setup(TRIG, GPIO.OUT)
    GPIO.setup(ECHO, GPIO.IN)

    GPIO.output(TRIG, False)
    os.system("espeak 'Calibrating Distance' ")
    # time.sleep(2)

    try:
        while True:
            #str = 'python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model'
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

                intDist = int(distance)
                l = len(str(intDist))
                new_dict = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                            6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 0: "Zero"}
                if (l == 3):
                    s1 = int(intDist/100)
                    s2 = int((intDist/10) % 10)
                    s3 = int(intDist % 10)
                    FinalString = new_dict[s1] + " " + \
                        new_dict[s2] + " " + new_dict[s3]

                elif (l == 2):
                    s1 = int(intDist/10)
                    s2 = int(intDist % 10)
                    FinalString = new_dict[s1] + " " + new_dict[s2]
                else:
                    FinalString = new_dict[intDist]

                os.system("espeak 'Object Detected at ' " +
                          FinalString + " ' metres ' ")
                # time.sleep(2)
                os.system("espeak 'Launching Camera for Object Detection' ")
                WebCam()
                # time.sleep(2)
                # os.system(str)

    except KeyboardInterrupt:
        GPIO.cleanup()


def WebCam():
    while True:

        t1 = cv2.getTickCount()

        frame1 = videostream.read()

        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(
            output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin),
                              (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                # Look up object name from "labels" array using class index
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(
                    scores[i]*100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10)
                # Draw white box to put label text in
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (
                    xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Draw label text

        # Draw framerate in corner of frame
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Object detector', frame)

        # Calculate framerate
        #t2 = cv2.getTickCount()
        #time1 = (t2-t1)/freq
        #frame_rate_calc = 1/time1
        if (object_name):
            os.system("espeak 'Approaching Object is " + label + " ' ")
        else:
            os.system("espeak 'Approaching object is unknown' ")

        # time.sleep(2)
        buzz()
        #os.system("python3 buzz.py")
        #os.system("espeak ' " +label+ " ' ")

        if cv2.waitKey(1) == ord('q'):
            break


buzz()
if (distance <= 150):
    WebCam()

# Clean up
cv2.destroyAllWindows()
videostream.stop()
