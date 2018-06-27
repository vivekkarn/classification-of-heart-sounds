#Importing required python libraries
import os
import librosa
import tensorflow as tf
import numpy as np

#Defining One-Hot as labels
normal_onehot = [1,0]
murmur_onehot = [0,1]

#Converting files in a folder into list of arrays containg the properties of the files
def decodeFolder(category):
	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

#Extracting the feataures of a wav file as inpurt to the data
def extract_feature(file_name):
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
	mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc = 20).T.flatten()[:, np.newaxis].T
	print(len(mfcc))
	return np.array(mfcc)

#train data
normal_sounds = decodeFolder("normal")
normal_labels = [normal_onehot for items in normal_sounds]
murmur_sounds = decodeFolder("murmur")
murmur_labels = [murmur_onehot for items in murmur_sounds]
train_sounds = np.concatenate((normal_sounds, murmur_sounds))
train_labels = np.concatenate((normal_labels, murmur_labels))

#test_data
test_sound = decodeFolder("test")

#####################TENSORFLOW#############################
#setting up hyperparameters
x = tf.placeholder(tf.float32,[None,193])
y_ = tf.placeholder(tf.float32,[None,2])
W = tf.Variable(tf.zeros([193,2]))
b = tf.Variable(tf.zeros([2]))
init = tf.global_variables_initializer()

#starting training process
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y+1e-9))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(260):
		sess.run(train_step,feed_dict={x:train_sounds, y_:train_labels})
	print("Training Done!")
	print(sess.run(y,feed_dict={x:test_sound}))
	#print(sess.run(tf.argmax(y,0),feed_dict={x:test_sound}))