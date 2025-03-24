#!/usr/bin/env python
# coding: utf-8

# In[94]:


# this tutorial is based on YouTuber sentdex's Convolutional Neural Network image classifier: 
    # part 1: https://www.youtube.com/watch?v=gT4F3HGYXf4 
    # part 2: https://www.youtube.com/watch?v=Ge65ukmJTzQ&t=342s
# the tutorial uses resources from a 2016/2017 Kaggle cometition here: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# we'll be using code from the following tutorial (created by the same YouTuber, sentdex): https://pythonprogramming.net/tflearn-machine-learning-tutorial/



# this import is for resizing the image
import cv2
# for arrays
import numpy as np
# for manipulating directories
import os
# for shuffling our data
from random import shuffle
# professional looping with progress bar
from tqdm import tqdm

import tensorflow as tf
import matplotlib.pyplot as plt

import random

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.compat.v1.reset_default_graph()

# for our tensorboard error
#import keras.backend as K

# Defining constants
# the directories to our training/testing data extractions
TRAIN_DIR = './train'
TEST_DIR = './test'
# the images are going to be 50x50, but not all the images are square, so there will be some distortion
IMG_SIZE = 50
# "Learning rate"
LR = 1e-3

# Model name so we can go back and tweak later
# saving the name with some formatting so we can keep track of the Learning Rate and the "2 convolutional layers"
# this will make it easier to plug into different neural networks later
MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '6conv-basic-video')


# In[95]:


# Now we want to load in the images and get features and labels 
# images are numbered as "cat.1", "dog.5" etc and they are color in color
# we want the images (the features) to be grayscale np arrays, which is a 2D array
# for labels, we to convert dog / cat to one-hot arrays


# In[96]:


# our one-hot array:
# first value is "dog", second value is "cat"
# so, a cat image is [0,1], a dog image is [1,0]

# for the "split" it works as follows: 
    # the images are named like "dog.93.png"
    # by splitting with the '.', you create 3 "indices" from the title where 'png' is -1, '93' is -2 and 'dog' is -3
    # so the code is saying that "word_label" is equal to the -3 index of the name, which will always be either dog or cat

# this function will take in every image in our training set, split the name, and return the one-hot encoded array for each image
# this transforms our strings of either "cat" or "dog" into a number representation (very cool)

def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [0,1]
    elif word_label == 'dog': return [1,0]


# In[97]:


# function to create the train data based on the label_img function

# this function will have the 25000 images with the features and the labels
# this is our training data
# later, we will test the accuracy of the training data by taking the 500 - 1000 training data images that were given to us to see how our training is going
# these testing images were given to us by Kaggle, so we will just pull them from the TEST_DIR path where we extracted them

# if you run into issues with the tqdm, just remove it and use the os.listdir(TRAIN_DIR) on it's own 

# training_data is initialized as an empty list
# for every image in our training set, we feed it to label_img() to get the one-hot array
# next, we want to process the images - we are going to resize the image, and then convert to grayscale using cv2

# cv2.imread() proccesses images taken from a path and returns it as a NumPy array - it takes 2 paramtere: the path to the file, and then the "flag"
    # the flag tells it to load either a color image (1), a grayscale image (0), or an unchanged image (-1)
    # you can pass the integer values as the flag, or pass cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, or cv2.IMREAD_UNCHANGED, respectively

# cv2.resize() takes two paramters here, the image and the target dimensions as a tuple

# at the end of the for loop, our processed image and the new label are then added to training_data as a new list (so each image is represented by a list of it's np array and it's one-hot encoded array)
    # here, we specify that they should be np arrays to make SURE that both the images and labels are in np array format
    # this is just an extra precaution since our cv2.imgread AND our cv2.resize() methods should have returned np arrays (even if the image wasn't loaded correctly in cv2.imread(), it will return an empty np array)

# lastly, we want to shuffle the data in the array, and SAVE OUR OUTPUT TO A NEW FILE
# saving our data to an output file is important because in the future, if we want to reuse any of this project, we don't have to re-process all of our images, we just need to reload our output file
# the only time we would need to re-process is if we use a new set of images or want to resize our grayscale images

# note: to get the np.save() to work, you have to set the training_data dtype to "object", as shown - when you load the file back in, make sure to write "allow_pickle = True" or you will have issues
# saving the entire np array as dtype = "object" will preserve the uint8 of the feature and label arrays, so there shouldn't be an issue to using this method in the next set of code
# if, for some reason, this causes an error, you will need to either:
    # NOT save to a .npy file (or any file, I tried with a .txt and got the same issues) and just run the code to create the data np array every time you want to use this program
    # figure out how to get numpy version 1.23.0 to work
# the issue is: we get an error trying to save the np array to the .npy file because of an "inhomogenous" array (ie the array is comprised of arrays of different dimensions) - however, the array is NOT inhomogenous
# I tried to find the "inhomogenous" arrays and I wasn't able to find anything (I added a line inside create_train_data() that would print the number of any instance that did not produce the correct shape for either img or label and nothing printed)
# I'm 99% sure that the array is not inhomogenous - it is ONLY a problem with saving the array
# this problem is a known issue, and the solutions online all say that version 1.23.0 is the most stable version to fix this issue - however, it is too old to work with current versions of pip (and potentially other libraries like tensorflow)

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        num_label = np.array(label)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        training_data.append([np.asarray(img), num_label])
    shuffle(training_data)
    np.save('train_data.npy', np.array(training_data, dtype="object"))
    return training_data
    
        


# In[98]:


# this function is essentially doing the same thing as the create_train_data() function, just on the images that were given to us to use as the testing images

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    np.save('test_data.npy', np.array(testing_data, dtype="object"))
    return testing_data


# In[99]:


# for these next lines, you need to pick how to get the training data
# you either call the "create_train_data()" to process the images for the first time
# OR if you already have the training data, you just load it from the saved file

# NOTE:
# make sure to COMMENT OUT the option you're not using !!!
# we could run checks to see if we already have the data file saved, but we would also then need to check whether everything was of the SAME DIMENSION
# in this case, we'll just manually comment/uncomment code

# if you're running images for the first time, uncomment the next line:
# train_data = create_train_data()

# if you've already run these images and have the saved training_data.npy file, uncomment the next line:
train_data = np.load('train_data.npy', allow_pickle=True)



# In[100]:


my_data = np.load('./train_data.npy', allow_pickle=True)


# In[101]:


# This cell is based on code here: https://pythonprogramming.net/tflearn-machine-learning-tutorial/
# the warnings that this cell produces when run are due to older versions of Pillow and tflearn that we needed to install to fix issues with code not being supported

# in case you need, it to fix the errors about tflearn, use this Stack Overflow article: https://stackoverflow.com/questions/76866418/error-when-i-try-to-import-tflearn-cannot-import-name-is-sequence-from-tens
    # to fix the main tflearn errors:
    # pip uninstall tflearn
    # pip install git+https://github.com/MihaMarkic/tflearn.git@fix/is_sequence_missing
    # to fix the ANTIALIAS errors:
    # pip uninstall Pillow
    # pip install Pillow==9.5.0

# Because this code was originally from the handwritten-digits MNIST, we will make some changes
# changing our input size to be the size our images

# ran into issues with model.fit() down the line, one solution said the "name" variables were the issue so it's been commmented out but not deleted, just in case
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1])

# recall: 1 layer is for linear problems, 2+ layers is for non-linear
# FOR 2 LAYERS: use only the following two conv_2d:
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# FOR 6 LAYERS: uncomment the following copy/pasted additional 4 layers:

# ***begin additional layers***
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


# ***end of additional layers***

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# the digit MNIST dataset had 10 types of examples, 0 - 9, but we only have 2, cats and dos
convnet = fully_connected(convnet, 2, activation='softmax')
# we will replace the previous learning rate with our defined learning rate value, LR
# ran into issues with model.fit() down the line, one solution said the "name" variables were the issue so it's been commmented out but not deleted, just in case
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='target')
# convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy')

# the second parameter wasn't necessary in the original code, so we will add the following second paramter:
    # tensorboard_dir='log'
# in other os, this would automatically log, but here we have to provide this instruction explicitly
model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[102]:


# before training the network, we write the following check:

# this is saying, if the meta file for our model already exists, you've saved a checkpoint ie you've already saved a number of epochs
# this means you can save your progress as you go
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


# In[103]:


# now we will separate out training and testing
# in theory, both our testing and training data should have the same accuracy

# this line says that our training data is all but the last 500 images
train = train_data[:-500]
# this line says that our testing data is only the last 500 images
test = train_data[-500:]


# In[104]:


# now we will process some of this to get ready for the tflearn code

# our training data (this is the data that is getting fit)
# X is our image/pixel data, and we will use this list comprehension to create the X list of all images
# Y is our label data
X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array([i[1] for i in train])

# this is for our test data (the data we are using to test accuracy)
test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = np.array([i[1] for i in test])


# In[105]:


# this is the code that will actually fit our data

# model.fit() was causing a lot of weird errors - the only solution that worked was to restart the kernel

# most of this code was copy/pasted from the same tutorial as the convent code above
# we'll just adjust this code to have only 5 epochs instead of 10
# we'll also change the run_id variable to our MODEL_NAME - this is how find our data in tensorboard

# NOTE: now that we can just load our model, we don't need to run this code
# if you DO need to re-create the model, or you want to re-train a loaded model, uncomment the next two lines:
#model.fit({'input': X}, {'target': Y}, n_epoch=5, validation_set=({'input': test_x}, {'target': test_y}), 
    #snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# In[106]:


# Using tensorboard!
# NOTE ABOUT TENSORBOARD: if you run the model again with the same name, it will write multiple files to the logs
# tensorboard will then try to give you files with "overwritten" graphs, but the numbers still look wonky
# make sure that you are periodically deleting models with the same name from the folder if you are running the program mulitple times without changing the name

# This next line will tell you which port your tensorboard is running on
# to use, copy the line below into a new terminal (OUTSIDE of jupyter) and hit enter
# tensorboard --logdir="C:\Users\MBartley\classification\log"
    # note: the tutorial says that you have to give it a name, but (boardname:C\Users...) but that did not work for me. The tutorial also makes it look like you don't need quotes, but you DO
    # maek sure to copy/paste it EXACTLY as it shows above
# now, copy/paste the port it gives you into a new browser window and you can view your logs
# on tensorboard, make sure you're on the "Scalars" tab on the top menu
# REMEMBER to either toggle y or click the "4 arrows" button to fit graph dimensions to the data
# smoothing also helps, under the settings tab on the right
# our numbers look a little different than the tutorial, I think this is due to improvements in tensorflow over the years in addition to just the variance in training


# In[107]:


# Now that we have our model trained, we can save it
# in the future, our program will load this model instead of creating a new one
# this is what our "os.path.exists()" condition was for
# with the model loaded, we can then re-train it by setting it run for another 5 epochs, for example
model.save(MODEL_NAME)


# In[109]:


# Testing our classifier!

# if you don't have test data yet, uncomment this line:
#test_data = process_test_data()

# if you do have test data, uncomment this line:
test_data = np.load('./test_data.npy', allow_pickle=True)

fig = plt.figure()

# now, we will iterate through our first 12 or so test data, plot them on a figure, and set the title to be the classification


start = random.randint(0, 488)
end = start + 12
for num, data in enumerate(test_data[start:end]):
    # cat: [0, 1]
    # dog: [1, 0]
    img_num = data[1]
    img_data = data[0]
    
    # generating our image graph
    y = fig.add_subplot(3, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    
    # this is how we call our model to classify
    # predict() takes a list as input, and returns a list of predictoins
    # since we just want to know the first, we set it equal to 0
    model_out = model.predict([data])[0]
    
    # here, we set the output to be either cat or dog
    if np.argmax(model_out) == 0: str_label = 'Dog'
    else: str_label = 'Cat'
    
    # now we say that we want to see the original image, in "gray" (still include this even though our images are already in grayscale)
    # then, we add our Cat or Dog prediction title
    # Then, we get rid of our subplot's axes to clean up the output and make it easier to read
    # lastly, we call plt.show() to generate the graph
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




