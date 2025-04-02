import tflite_runtime.interpreter as tflite
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import pickle
import face_recognition
import matplotlib.pyplot as plt

# Define VideoStream class for video capture
class VideoStream:
    """Camera object that controls video streaming"""
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
parser.add_argument('--labels', help='Path to label map', default='models/labels.txt')
parser.add_argument('--threshold', help='Confidence threshold for object detection', type=float, default=0.5)
parser.add_argument('--resolution', help='Video resolution WxH', default='640x480')
parser.add_argument('--face_encodings', help='Path to face encodings file', required=True)
args = parser.parse_args()

# Load TFLite object detection model
interpreter = tflite.Interpreter(model_path=args.model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Load label map
with open(args.labels, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load face encodings
with open(args.face_encodings, "rb") as f:
    face_data = pickle.load(f)
known_face_encodings = face_data["encodings"]
known_face_names = face_data["names"]

# Parse resolution
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Start video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Set up video writer with a unique filename
timestamp = int(time.time())
video_filename = f"output_{timestamp}.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, 20.0, (imW, imH))
print(f"Recording video to {video_filename}")

# Data for plotting
timestamps = []
detected_names = []
detected_objects = []
confidence_levels = []

# Start time
start_time = time.time()

# Process video frames
while True:
    t1 = cv2.getTickCount()
    frame1 = videostream.read()

    # Write the frame to the video file
    out.write(frame1)

    # Convert frame to RGB for face recognition and object detection
    frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(cv2.resize(frame_rgb, (width, height)), axis=0)

    # Normalize input if model requires floating-point data
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform object detection
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw object detection results
    current_objects = []
    current_confidences = []
    for i in range(len(scores)):
        if scores[i] > args.threshold:
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            object_name = labels[int(classes[i])]
            label = f"{object_name}: {int(scores[i] * 100)}%"
            current_objects.append(object_name)
            current_confidences.append(scores[i])
            cv2.rectangle(frame1, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(frame1, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Perform face recognition
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    current_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        current_names.append(name)
        current_confidences.append(1 - face_distances[best_match_index])  # Confidence level
        cv2.rectangle(frame1, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame1, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Store data for plotting
    timestamps.append(time.time() - start_time)
    detected_names.append(current_names)
    detected_objects.append(current_objects)
    confidence_levels.append(current_confidences)

    # Display FPS on frame
    t2 = cv2.getTickCount()
    frame_rate_calc = cv2.getTickFrequency() / (t2 - t1)
    cv2.putText(frame1, f"FPS: {frame_rate_calc:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show video stream
    cv2.imshow('Object and Face Detection', frame1)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping video processing as 'q' was pressed.")
        break

# Cleanup
out.release()
cv2.destroyAllWindows()
videostream.stop()

# Save the plot as an image
plot_filename = f"output_plot_{timestamp}.png"
plt.figure(figsize=(12, 8))

# Detected Names
plt.subplot(3, 1, 1)
for i, names in enumerate(detected_names):
    for name in names:
        plt.scatter([timestamps[i]], [name], label=name if i == 0 else "", color='blue')
plt.title("Detected Names Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Names")

# Detected Objects
plt.subplot(3, 1, 2)
for i, objects in enumerate(detected_objects):
    for obj in objects:
        plt.scatter([timestamps[i]], [obj], label=obj if i == 0 else "", color='green')
plt.title("Detected Objects Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Objects")

# Confidence Levels
plt.subplot(3, 1, 3)
for i, confidences in enumerate(confidence_levels):
    for conf in confidences:
        plt.scatter([timestamps[i]], [conf], color='red')
plt.title("Confidence Levels Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Confidence")

plt.tight_layout()
plt.legend()
plt.savefig(plot_filename)  # Save the plot as a PNG file
plt.show()
print(f"Plot saved as {plot_filename}")
