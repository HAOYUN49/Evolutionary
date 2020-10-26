import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import os
import urllib.request
import pickle
import gzip

def extract_data(filename, num_images):
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(num_images*28*28)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = (data / 255) - 0.5
		data = data.reshape(num_images, 28, 28, 1)
	return data

def extract_labels(filename, num_images):
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1*num_images)
		labels = np.frombuffer(buf, dtype=np.uint8)
	return (np.arange(10) == labels[:, None]).astype(np.float32)



class MNIST:
	def __init__(self):
		if not os.path.exists("data"):
			os.mkdir("data")
			files = ["trian-images-idx3-ubyte.gz",
					"t10k-images-idx3-ubyte.gz",
					"train-labels-idx1-ubyte.gz",
					"t10k-labels-idx1-ubyte.gz"]
			for name in files:
				urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

		train_data = extract_data("data/trian-images-idx3-ubyte.gz", 60000)
		train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
		self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
		self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)

		VALIDATION_SIZE = 5000

		self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
		self.validation_labels = train_labels[:VALIDATION_SIZE, :, :, :]
		self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
		self.train_labels = train_labels[VALIDATION_SIZE:, :, :, :]

class  MNISTModel:
	def __init__(self, restore = None, uss_log=False):
		self.num_channels = 1
		self.image_size = 28
		self.num_labels = 10

		model = Sequential()

		model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
		model.add(Activation('relu'))
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(200))
		model.add(Activation('relu'))
		model.add(Dense(200))
		model.add(Activation('relu'))
		model.add(Dense(10))

		if restore:
			model.load_weights(restore)

		self.model = model

		def predict(self, data):
			return self.model(data)
