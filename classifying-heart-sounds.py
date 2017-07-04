import os
import librosa
import tensorflow as tf
import numpy as np


normal_onehot = [1,0,0]
murmur_onehot = [0,1,0]
extrasystole_onehot = [0,0,1]

def decodeFolder(category):
	print("Starting decoding folder...")
	listOfFiles = os.listdir(category)
	arrays_sound = np.empty((0,193))
	for file in listOfFiles:
		filename = os.path.join(category,file)
		features_sound = extract_feature(filename)
		arrays_sound = np.vstack((arrays_sound,features_sound))
	return arrays_sound

def extract_feature(file_name):
	print("Extracting...")
	X, sample_rate = librosa.load(file_name)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
	return np.hstack((mfccs,chroma,mel,contrast,tonnetz))


murmur_sounds = decodeFolder("murmur")
print(len(murmur_sounds))
#extrasystole_sounds = decodeFolder("extrasystole")
#normal_sounds = decodeFolder("normal")
