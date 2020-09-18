import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.kreas.prepocessing import image
from PIL import Image
import pickle
import time
import datetime
import os
import argparse

#实现 SphereFace, CosFace, ArcFace
#trian models
#处理图片，正确的label
#准备untargeted targeted不同的adversarial example和example
#根据paper进行attack
#存储图片

#doging attack
#impersonation attack
def evolutionary_attack(args):
	covari_matri = np.identity(45)
	evo_path = 0
	cc = 0.01
	ccov = 0.001
	succ_rate = 
	distance = 0


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--save", default="./saved_results")
	parser.add_argument("-n", "--numing", type=int, default=0, help="number of test images to attack")
	parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to default value")
	parser.add_argument("-f", "--firstimg", type=int, default=0, help="number of image to start with")
	parser.add_argument("-u", "--untargeted", action='store_true')
	parser.add_argument("-p", "--propotional", action='store_true')
	args = vars(parser.parse_args())

	if arg['maxiter'] == 0:
		if arg['untargeted']:
			arg['maxiter'] =
		else:
			arg['maxiter'] =

	evolutionary_attack(args)