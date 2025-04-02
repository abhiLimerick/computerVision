import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import csv
import face_recognition

# Define VideoStream class to handle streaming of video from the PiCamera
class VideoStream:
    """Camera object that controls video streaming from the PiCamera"""
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])
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


# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to the TFLite model', default='models/ssdmobilenet.tflite')
parser.add_argument('--labels', help='Path to the Labels', default='models/labels.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects', default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution (WxH)', default='1280x720')
args = parser.parse_args()

# Load model and labels
PATH_TO_MODEL_DIR = args.model
PATH_TO_LABELS = args.labels
MIN_CONF_THRESH = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path=PATH_TO_MODEL_DIR)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Load labels
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Variables for logging and analysis
log_data = []
people_in_scene = {}  # Track people and confidence
entry_exit_times = {}  # Track entry, exit times, and max confidence

# Generate a unique filename using the current timestamp
timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())  # Format: YYYYMMDD_HHMMSS
output_video_filename = f"processed_video_{timestamp}.avi"

# Initialize video writer to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use XVID codec
video_writer = cv2.VideoWriter(output_video_filename, fourcc, 30, (imW, imH))

# Start the timer for 20 seconds
start_time = time.time()

while time.time() - start_time < 20:
    current_count = 0
    t1 = cv2.getTickCount()

    # Grab frame from the video stream
    frame1 = videostream.read()
    frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform object detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Loop over all detections and draw detection box if confidence is above threshold
    for i in range(len(scores)):
        if scores[i] > MIN_CONF_THRESH:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            confidence = scores[i]  # Current confidence score
            label = '%s: %d%%' % (object_name, int(confidence * 100))

            # Draw bounding box and label
            cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame1, (xmin, label_ymin - labelSize[1] - 10), 
                          (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame1, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Log the detected object with confidence
            log_data.append([time.time(), object_name, confidence, "object"])
            people_in_scene[object_name] = confidence

            # Track entry and update confidence
            if object_name not in entry_exit_times:
                entry_exit_times[object_name] = {
                    'entry_time': time.time(),
                    'exit_time': None,
                    'max_confidence': confidence
                }
            else:
                # Update the maximum confidence score
                entry_exit_times[object_name]['max_confidence'] = max(
                    entry_exit_times[object_name]['max_confidence'], confidence
                )

            current_count += 1

    # Update exit times for objects no longer detected
    detected_objects = [labels[int(classes[i])] for i in range(len(scores)) if scores[i] > MIN_CONF_THRESH]
    for name in list(entry_exit_times.keys()):
        if name not in detected_objects and entry_exit_times[name]['exit_time'] is None:
            entry_exit_times[name]['exit_time'] = time.time()

    # Calculate and display frame rate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Display frame with FPS and detection count
    cv2.putText(frame1, 'FPS: {0:.2f}'.format(frame_rate_calc), (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 55), 2)
    cv2.putText(frame1, 'Total Detection Count: ' + str(current_count), (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 55), 2)

    # Show object presence and confidence over time
    y_offset = 100  # Starting Y position for text display
    for name, confidence in people_in_scene.items():
        cv2.putText(frame1, f"{name}: {confidence*100:.2f}%", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30  # Space between text

    # Display the frame
    cv2.imshow('Object Detector', frame1)

    # Write the processed frame to the video file
    video_writer.write(frame1)

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
video_writer.release()  # Release the video writer to save the file

# Save log data for analysis
with open("log_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object Name", "Confidence", "Type"])
    writer.writerows(log_data)

print("[INFO] Data saved to log_data.csv")
print(f"[INFO] Processed video saved as {output_video_filename}")

# Post-Event Analysis: Print Entry/Exit Times and Confidence
print("\nEntry/Exit Times and Confidence:")
for name, times in entry_exit_times.items():
    exit_time = f"{times['exit_time']:.2f}" if times['exit_time'] is not None else "Still in scene"
    print(f"{name} entered at {times['entry_time']:.2f}, exited at {exit_time}, "
          f"max confidence: {times['max_confidence']*100:.2f}%")


print("Done")