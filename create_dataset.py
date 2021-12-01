# This file will detect face ROIs and use it to build a dataset for livenessNet
import numpy as np 
import argparse 
import cv2
import os

# Load the face detector model
protoPath = "./face_detector/deploy.prototxt"
modelPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Open video stream
vs = cv2.VideoCapture("./videos/real.MOV")
read = 0	# Number of frames to read
saved = 0	# Number of frames saved while our loop executes

# Skip rate
skip_rate = 4

# Loop over the frames in the video
while True:
	# get a frame
	(grabbed, frame) = vs.read()

	# If there is no frame grabbed => we've reached end of file
	if not grabbed:
		break

	# Increase the total number of frames read so far
	read += 1

	# If this frame is within the skip range => continue
	if read % skip_rate != 0:
		continue

	# Construct a blob from frame
	(h, w) = frame.shape[:2]
	# This blob has 300x300 dim to accommodate the face detector model
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

	# Pass blob to face detector & obtain detections
	net.setInput(blob)
	detections = net.forward()

	# Ensure at least 1 face was found
	if len(detections) > 0:
		# Because all frames have only 1 face in it => find index of the bounding box with highest probability
		i = np.argmax(detections[0,0,:,2])
		# Extract confidence of the found index
		confidence = detections[0,0,i,2]

		# Filter out weak predictions
		if confidence > 0.5:
			# Comput (x, y) - the coordinate of bounding box & extract face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			face = frame[startY:endY, startX:endX]

			# Write frame to the disk
			p = "./dataset/real/{}.png".format(str(saved) + "_v2")
			cv2.imwrite(p, face)
			saved += 1

# Cleanup
vs.release()
cv2.destroyAllWindows()