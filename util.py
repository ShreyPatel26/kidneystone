import tensorflow as tf
import numpy as np
import io

MODEL = tf.keras.models.load_model("kidney_stone_model.h5")
CLASSES = [
	"Congratulations! The patient is HEALTHY",
	"KIDNEY STONE DETECTED, Seek medical help!!"
]

def load_img(img):
	# Reading The Image
	img = img.file.read()
	# Converting to Bytes Stream
	img = io.BytesIO(img)
	# Loading Image
	img = tf.keras.preprocessing.image.load_img(img, target_size=MODEL.input_shape[1:])
	# Converting Image to Array Of Correct Dimensions
	img = np.expand_dims(img, 0)
	# Returning Loaded Image
	return img

def preprocess(img):

	# img = tf.keras.applications.vgg16.preprocess_input(img)
	img = img / 255.

	return img

def predict(X):

	if MODEL.output_shape[1] == 1:
		# last layer activation used
		pred = float(MODEL.predict(X)[0, 0])
	else:
		# last layer softmax used
		pred = float(MODEL.predict(X)[0, 1])

	i = round(pred)

	if i == 0:
		pred = 1 - pred

	return {
		"prediction": CLASSES[i],
		"probability": pred
	}

	"""
	pred = MODEL.predict(X)[0, 0]
	i = np.argmax(pred[0])
	return {
		"prediction": CLASSES[i],
		"probability": round(pred[0, i].tolist(), 3)
	}
	"""

def pipeline(img):

	loaded_img = load_img(img)
	processed_img = preprocess(loaded_img)
	response = predict(processed_img)

	return response