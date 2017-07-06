import os
import librosa
import tensorflow as tf
import numpy as np


normal_onehot = [1,0,0]
murmur_onehot = [0,1,0]
extrasystole_onehot = [0,0,1]

def decodeFolder(category):
	print("Starting decoding folder "+category+" ...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

def extract_feature(file_name):
	print("Extracting "+file_name+" ...")
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return np.hstack((mfccs,chroma,mel,contrast,tonnetz))

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
y_ = tf.placeholder(tf.float32,[None,3])
W = tf.Variable(tf.zeros([193,3]))
b = tf.Variable(tf.zeros([3]))
init = tf.global_variables_initializer()

#starting training process
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(200):
		sess.run(train_step,feed_dict={x:train_sounds, y_:train_labels})
	print("Training Done!")
	print(sess.run(y,feed_dict={x:test_sound}))
