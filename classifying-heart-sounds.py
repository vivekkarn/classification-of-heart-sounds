#Importing all the required documents

import os
import librosa
import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

#Setting up

murmurs = os.listdir("murmur")
extrasystole = os.listdir("extrasystole")
normal = os.listdir("normal")

#Initialiazing arrays

murmur_sounds = []
extrasystole_sounds = []
normal_sounds = []
for file in murmurs:
    y,sr = librosa.load(os.path.join("murmur",file))
    murmur_sounds.append(y)
for afile in extrasystole:
    y1,sr1 = librosa.load(os.path.join("extrasystole",afile))
    extrasystole_sounds.append(y1)
for bfile in normal:
    y2,sr2 = librosa.load(os.path.join("normal",bfile))
    normal_sounds.append(y2)

#Waah

n_classes = 3
x = tf.placeholder(tf.float32,[None,None])
Y = tf.placeholder(tf.float32,[None,3])
W = tf.Variable(tf.zeros())
b = tf.Variable(tf.zeros(3))
sess = tf.Session()
pred = tf.nn.softmax(tf.matmul(x, W) + b)