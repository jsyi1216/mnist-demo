from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from keras.models import model_from_json
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import itertools

sess = tf.Session()
sess.run(tf.global_variables_initializer())
a=[1,2,3,4,5]
b=tf.expand_dims(a,0).eval(session=sess)
c=tf.expand_dims(b,2).eval(session=sess)
# filename = 'prediction_Module/trained_models/nn_trained_digit_classifier.pkl'
modelpath = 'prediction_Module/trained_models/'

def predict_nn(img):
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	model = load_model(modelpath+"model.hdf5")
	img=tf.expand_dims(img,0).eval(session=sess)
	img=tf.expand_dims(img,3).eval(session=sess)
	label = model.predict_classes(img)[0]

	return label


def preprocess(image_path):
	img = Image.open(image_path)
	img = np.asarray(img.getdata(), dtype = 'float32')
	img = img / 255.

	if len(img.shape) > 1:
		img = np.delete(img, np.s_[1:], 1)
		img = img.flatten()
	if img.shape[0] != 28 or img.shape[1] != 28:
		img = np.resize(img, (28,28))

	return img


# This function will be called from Django website with argument as the image path
def predict(image_path, clf='NN'):

	img = preprocess(image_path)

	if clf == 'NN':
		# img = img.flatten()
		label = predict_nn(img)

	return label    # The output should be a json with "name of the file, label predicted and confidence value. "


def loadModel():
	# Load the architecture of model
	json_file = open(modelpath+"model.json","r")
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	
	# Load the weights
	model.load_weights(modelpath+"model.h5")

	return model