!unzip dataset.zip

# set the matplotlib backend so figures can be saved in the background
import matplotlib		# To generate training plot
matplotlib.use("Agg")	# Specify "Agg" backend so we can save the plots to disk

# import the necessary packages
from pyimagesearch.livenessnet import LivenessNet 		# The LivenessNet CNN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split	# Splits the data for training & testing
from sklearn.metrics import classification_report		# Generate statistical report of model's performance
from tensorflow.keras.preprocessing.image import ImageDataGenerator	  # For data augmentation, provide us with batches of randomly mutated images
from tensorflow.keras.optimizers import Adam 			# Adam optimizer
from tensorflow.keras.utils import to_categorical		
from imutils import paths								# Help to gather paths to all the image files on disk 
import matplotlib.pyplot as plt 						# Generate training plot
import numpy as np 										
import argparse 										# Processing command line arguments 
import pickle											# Serialize our label encoder to disk
import cv2
import os

# Initial learning rate, batch size, epochs
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

# Load images in the dataset directory. Initialise data & labels list
print("[INFO] loading images...")
imagePaths = list(paths.list_images("dataset"))
data = []
labels = []

# Loop over all image paths
for imagePath in imagePaths:
	'''
	- extract the label from the filename
	- load the image
	- resize it to be 32x32 pixels, ignoring aspect ratio
	'''
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32))

	# Update the data and labels lists
	data.append(image)
	labels.append(label)

# convert images into a NumPy array, and scale all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers, then one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# splits the data into training and testing using 75% for training and 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# construct a data augmentation object -> generate images with random rotations, shifts, ...
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32, height=32, depth=3,
	classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit(x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

