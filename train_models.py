import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

from setup_mnist import MNIST 
from setup_cifar import CIFAR 

import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
	model = Sequential()

	model.add(Conv2D(params[0], (3, 3), input_shape=data.train_data.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(params[1], (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(params[2], (3, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(params[3], (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(params[4]))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(params[5]))
	model.add(Activation('relu'))
	model.add(Dense(10))

	if init != None:
		model.load_weights(init)

	else:
		def fn(correct, predicted):
			return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
														logits=predicted/train_temp)

		sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

		model.compile(loss=fn,
					optimizer=sgd,
					metrics=['accuracy'])

		model.fit(data.train_data, data.train_labels,
				batch_size=batch_size,
				epochs=num_epochs,
				validation_data=(data.validation_data, data.validation_labels),
				shuffle=True)

	if file_name != None:
		model.save(file_name)

	return model

if not os.path.isdir('models'):
	os.makedirs('models')

train(CIFAR(), 'models/cifar.h5', [64, 64, 128, 128, 256, 256], num_epochs=50)
train(MNIST(), 'models/mnist.h5', [32, 32, 64, 64, 200, 200], num_epochs=50)

