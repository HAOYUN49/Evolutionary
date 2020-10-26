#for mnist & cifar
#no stochastic coordinate selection
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.kreas.prepocessing import image
from PIL import Image
import pickle
import time
import random
import datetime
import os
import argparse
from math import exp 

from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel


#实现 SphereFace, CosFace, ArcFace
#trian models
#处理图片，正确的label
#准备untargeted targeted不同的adversarial example和example
#根据paper进行attack
#存储图片

#doging attack
#impersonation attack

def show(img, name = "output.png"):
	fig  = (img + 0.5)*255
	fig = fig.astype(np.uint8).squeeze()
	pic = Image.fromarray(fig)
	pic.save(name)

def evolutionary_attack(args, adversarial, original, ori_label, adv_label):
	
	n =  np.prod(adversarial.shape) # the dimension of the input space
	m =  np.prod(adversarial.shape) # dimension of the search space
	k =  np.prod(adversarial.shape) # the number of coordinates for stochastic coordinate selection

	covari_matri = np.eye(m)
	evo_path = np.zeros(m) #pc
	zero_matrix = np.zeros(m)
	cc = 0.01
	ccov = 0.001
	succ = 0
	uu = 1
	sigma = 0.01 * ((np.sum((adversarial-original)**2))**.5)
	maxiter = args['maxiter']
	for i in range(maxiter):
		#normalize the images？？
		perturb_random = np.array(np.random.multivariate_normal(zero_matrix, (sigma**2)*covari_matri))
		perturbation = np.array(perturb_random + uu*(original - adversarial))
		perturbation = np.clip(perturbation, 0, 1)
		children = np.array(perturbation.reshape(adversarial.shape) + adversarial)
		pred_r = model.model.predict(children)

		is_success = True
		if args['targeted']:
			if np.argmax(pred_r, 1) != np.argmax(adv_label, 1):
				is_success = False
		else:
			if np.argmax(pred_r, 1) == np.argmax(ori_label, 1):
				is_success = False

		if not is_success:
			uu *= exp(succ/i - 1/5)
			continue
		else:
			children_dist = (np.sum((children-original)**2))**.5
			parent_dist = (np.sum((adversarial-original)**2))**.5
			if children_dist < parent_dist:
				succ += 1
				sigma = 0.01 * children_dist
				adversarial = children
				uu *= exp(succ/i - 1/5)
				evo_path = (1-cc)*evo_path + (((cc*(2-cc))**.5)/sigma)*perturb_random
				for j in range(m):
					covari_matri[j][j] = (1-ccov)*covari_matri[j][j] + ccov*((evo_path[j])**2)
			else:
				uu *= exp(succ/i - 1/5)
				continue



def generate_data(data, model, samples, targeted=False, start=0, seed = 3):
	
	random.seed(seed)
	inputs = []
	targets = []
	adv_labels = []
	ori_labels = []
	true_ids = []

	for i in range(samples):
		
		target = data.test_data[start+i]
		ori_label = data.test_labels[start+i]
		if targeted:
			seq = np.random.randint(data.test_labels.shape[0])
			adversarial = data.test_data[seq]
			while True:
				pred_r = model.model.predict(adversarial)
				if (np.argmax(pred_r, 1) != np.argmax(ori_label, 1)):
					break
				seq = np.random.randint(data.test_labels.shape[0])
				adversarial = data.test_data[seq]
			adv_label=np.eye(data.test_labels.shape[1][np.argmax(pred_r, 1)])
			
		# random adversarial, close to targets, not equal to slabel	
		else:
			#inputs(random but adversarial)
			adversarial = np.random.random(target.shape)
			while True:
				pred_r = model.model.predict(adversarial)
				if (np.argmax(pred_r, 1) != np.argmax(label, 1)):
					break
				adversarial = np.random.random(target.shape)
			adv_label=np.eye(data.test_labels.shape[1][np.argmax(pred_r, 1)])

		inputs.append(adversarial)
		targets.append(target)
		ori_labels.append(ori_label)
		adv_labels.append(adv_labels)
		true_ids.append(start+i)

	inputs = np.array(inputs)
	targets = np.array(targets)
	ori_labels = np.array(ori_labels)
	adv_labels = np.array(adv_labels)
	true_ids = np.array(true_ids)
	return inputs, targets, adv_labels, ori_labels, true_ids

def main(args):
	
	print("Loading model", args['dataset'])

	if args['dataset'] == "mnist":
		data, model = MNIST(), MNISTModel("models/mnist.h5")
	elif args['dataset'] == "cifar10":
		data, model = CIFAR(), CIFARModel("models/cifar.h5")

	print("Done...")

	if args['numing'] == 0:
		args['numing'] = len(data.test_labels) - args['firstimg']

	print("Using", args['numing'], "test images.")

	print("Generate data")

	all_inputs, all_targets, all_adv_labels, all_ori_labels, all_true_ids = generate_data(data, model, samples=args['numing'], targeted=args['targeted'],
																		start=args['firstimg'], seed=args['seed'])
	print("Done...")
	os.system("mkdir -p {}/{}".format(args[save], args['dataset']))
	img_no = firstimg
	MSE_total = .0 #tf.keras.losses.MSE(a, b)

	for i in range(all_true_ids.size):
		adversarial = all_inputs[i:i+1]
		original = all_targets[i:i+1]
		ori_label = all_ori_labels[i:i+1]
		adv_label = all_adv_labels[i:i+1]
		
		#test whether the image is correctly classified
		true_label = np.argmax(ori_label, 1)
		print("true labels:", true_label)
		original_predict = model.model.predict(original)
		predicted_class =  np.argmax(original_predict, 1)
		print("origial classification:", predicted_class)
		if (true_label != predicted_class):
			print("Skip wrongly classified image no. {}, original class {}, classified as {}".format(true_ids[i], true_label, predicted_class))
			continue

		img_no += 1
		timestart = time.time()
		adv =  evolutionary_attack(args, adversarial, original, ori_label, adv_label)
		timeend = time.time()
		MSE = tf.keras.losses.MSE(adv, original)
		MSE_total += MSE
		adversarial_predict = model.model.predict(adv)
		adversarial_class = np.argmax(adversarial_predict, 1)
		print("adversarial classification:", adversarial_class)
		suffix = "id{}_prev{}_adv{}_dist(MSE){}".format(all_true_ids[i], predicted_class, adversarial_class, MSE)
		print("Saving to", suffix)
		show(original, "{}/{}/{}/{}_original_{}.png".format(args['save'], args["dataset"], args["attack"], img_no, suffix))
		show(adv, "{}/{}/{}/{}_adversarial_{}.png".format(args['save'], args["dataset"], args["attack"], img_no, suffix))
		show(adv - original, "{}/{}/{}/{}_diff_{}.png".format(args['save'], args['dataset'], args['attack'], img_no, suffix))
		print("[STATS][L1] total = {}, id = {}, time = {:.3f}, prev_class = {}, new_class = {}, distortion(MSE) = {:.5f}, average MSE: {:.5f}".format(img_no, all_true_ids[i], timeend - timestart, predicted_class, adversarial_class, MSE, MSE_total/img_no))
		sys.stdout.flush()


			



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", choices=["mnist", "cifar10"], default="mnist")
	parser.add_argument("-s", "--save", default="./saved_results")
	parser.add_argument("-n", "--numing", type=int, default=0, help="number of test images to attack")
	parser.add_argument("-m", "--maxiter", type=int, default=0, help="set 0 to default value")
	parser.add_argument("-f", "--firstimg", type=int, default=0, help="number of image to start with")
	parser.add_argument("-ta", "--targeted", default = False, action="store_true")
	parser.add_argument("-sd", "--seed", type=int, default = 373, action="seed for generating random number")
	#parser.add_argument("-p", "--propotional", action='store_true')
	args = vars(parser.parse_args())
	if args['targeted']:
		args['attack'] = "targeted"
	else:
		args["attack"] = "untargeted" 

	if args['maxiter'] == 0:
		if args['dataset'] == "mnist":
			arg['maxiter'] = 3000
		elif args['dataset'] == "cifar10":
			arg['maxiter'] = 1000

	print(args)
	main(args)
	#evolutionary_attack(args)