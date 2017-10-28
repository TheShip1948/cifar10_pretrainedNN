##################################################
# Using pre-trained NN to predict images 
# from cifar10 dataset 
##################################################

##################################################
# --- Importing --- 
##################################################
# import the necessary packages
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow ONLY
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
# Not able to download opencv library
# import cv2

##################################################
# --- Construct and parse arguments --- 
##################################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="name of pre-trained network to use")
args = vars(ap.parse_args())

##################################################
# --- Dictionary definition --- 
##################################################
# define a dictionary that maps model names to their classes
# inside Keras
MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # TensorFlow ONLY
	"resnet": ResNet50
}
 
# esnure a valid model name was supplied via command line argument
if args["model"] not in MODELS.keys():
	raise AssertionError("The --model command line argument should "
		"be a key in the `MODELS` dictionary")

##################################################
# --- Initialize input shape --- 
##################################################
# initialize the input image shape (224x224 pixels) along with
# the pre-processing function (this might need to be changed
# based on which model we use to classify our image)
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input
 
# if we are using the InceptionV3 or Xception networks, then we
# need to set the input shape to (299x299) [rather than (224x224)]
# and use a different image processing function
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

##################################################
# --- Initialize input shape --- 
##################################################
# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")


##################################################
# --- Load input image --- 
##################################################
# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
print("[INFO] loading and pre-processing image...")

# This is a workaround since the script expect image 
# on disk, but my images are loaded from keras 
# sol: save the image to the disk and then re-read it
from keras.datasets import cifar10
(X_train, y_train),(X_test, y_test) = cifar10.load_data()
test_example = 1
image = X_test[test_example]

import scipy.misc 
scipy.misc.imsave("image.jpg", image)

image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)
 
# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through thenetwork
image = np.expand_dims(image, axis=0)
 
# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
image = preprocess(image)


##################################################
# --- Classify image --- 
##################################################
# classify the image
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)
 
# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

CIFAR_LABELS = {
	0: "airplane",
	1: "automobile",
	2: "bird",
	3: "cat", 
	4: "deer", 
	5: "dog", 
	6: "frog", 
	7: "horse", 
	8: "ship", 
	9: "truck", 	
}
value = y_test[test_example].astype(int)
print(type (value))
# print (type(y_test[test_example].astype(int)))
# image_label = CIFAR_LABELS[y_test[test_example].astype(int)]
image_label = CIFAR_LABELS[8]
print ("Label: {}".format(image_label))


