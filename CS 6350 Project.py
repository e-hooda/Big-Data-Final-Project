# Databricks notebook source
pip install tensorflow

# COMMAND ----------

pip install nltk

# COMMAND ----------

# From the Keras.io docs, code for using preprocessing layers APIs for image augmentation.
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

imageAugmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1), 
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="imageAugmentation",
)

# COMMAND ----------

import os
from pickle import dump
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from keras.layers import InputLayer, Activation
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from tensorflow.keras.optimizers import Adam 

# Image size required for EfficientNet
IMG_SIZE = 224

# extract features from each image in the directory
def extractFeatures(directory):

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = imageAugmentation(inputs)

# This option excludes the final Dense layer that turns 1280 features on the penultimate layer into prediction of the 1000 ImageNet classes. 
# Replacing the top layer with custom layers allows using EfficientNet as a feature extractor in a transfer learning workflow.

    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
  
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print(model.summary())
    features = dict()
    trainImagesList = os.listdir(directory)
    for name in trainImagesList:
        filename = directory + name
        print(filename)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features

directory = "/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/flickr30k_images/flickr30k_images/"
features = extractFeatures(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/features.pkl', 'wb'))

# COMMAND ----------

from pickle import dump
photoFeats = pd.read_pickle('/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/features.pkl')

# Preserve key,value ordering
keys = list(photoFeats.keys())
values = list(photoFeats.values())
values = [x[0] for x in values] #each value is a list of 1 item

# GlobalAveragePooling2D significantly decreases the size of each featurized image
imgInput = tf.keras.layers.Input(shape=(7,7,1280)) # shape (7,7,1280) from EfficiencyNetB0
imgAvgPool = tf.keras.layers.GlobalAveragePooling2D()(imgInput) # shape (1280) output
poolModel = tf.keras.Model(inputs=[imgInput], outputs=[imgAvgPool]) 
# Apply the pooling
valuesTensor = tf.convert_to_tensor(values, dtype=tf.float32)
newValues = poolmodel.predict(valuesTensor)

# Save the new features to disk
featuresPooled =  dict(zip(keys, newValues))
dump(featuresPooled, open('/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/featuresPooled.pkl', 'wb'))

# COMMAND ----------

# Maps the list of images to the appropiate captions
def imagesMapCaption(train_images_list, train_captions):
    mappings = {}
    for i in train_images_list:
      caps = []
      captions = train_captions[train_captions['image_name'] == i]['comment']
      for caption in captions:
        caps.append(caption)
      mappings[i] = caps
    return mappings

# COMMAND ----------

import string
from string import digits

# Does preprocessing and removes punctuation and numerics
def formatCaptions(captionMappings):
  for image in captionMappings:
    imageCaptions = captionMappings[image]
#     print(imageCaptions)
    fCaptions = []
    for c in imageCaptions:
      print("c", c)
      captionNoNumeric = c.translate(str.maketrans('', '', digits)).split(" ")
      captionFormat = " ".join(list(filter(lambda x : len(x) > 1, captionNoNumeric)))
#       print(captionNoNumeric)
      caption = captionFormat.lower().strip()
      captionNoPunct = caption.translate(str.maketrans('', '', string.punctuation))
      captionSeq = "<begofseq> " + captionNoPunct + " <endofseq>"
      fCaptions.append(captionSeq)
    captionMappings[image] = fCaptions
  return captionMappings

# COMMAND ----------

import tensorflow as tf
import numpy as np

#
# LSTM
#



# Gate equations adapted from wikipedia definition of LSTM.
# Intuitive purpose of gates adapted from colah's LSTM blog.

#f forget    = sigma(W_f*x  +  U_f*ouput_prev  +  b_f)    :decides how much of cell_state to keep/forget
#i input     = sigma(W_i*x  +  U_i*ouput_prev  +  b_i)    :decides which new candidate values to remember
#s select    = sigma(W_s*x  +  U_s*ouput_prev  +  b_s)    :decides what portions of the cell_state to output
#c candidate =  tanh(W_c*x  +  U_c*ouput_prev  +  b_c)    :creates new candidate values to be added to cell_state
#cell_state = (forget*cell_prev)+(input*candidate) 
#output = select*tanh(cell_state)

# Standard notation seems to use "output" instead of "select".
# I decided to use "select" because this intermediary gate's function is not to BE the output,
# but to instead select which portions of the cell_state to output.




# Using the keras API to write the LSTM layer so that it works well with the rest of the network.
# Particularly taking instruction from keras's MinimalRNNCell example class.
class LSTM_Cell(tf.keras.layers.Layer):
    # For simplicity, the LSTM cell receives as input the 128 length text/linguistic feature vector.
    # Any intermediate results of the LSTM are also 128 length, as well as the output. (Assuming called with size=128)

    def __init__(self, size, **kwargs):
        self.size = size
        self.state_size = [self.size, self.size] # two 128-length vectors, not a 128x128 matrix
        self.output_size = self.size
        super(LSTM_Cell, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'size': self.size,
        })
        return config
    
    def predict(self, X):
        return self.model.predict(X)
    
    def build(self, input_shape):
        

        # Weights (matrix) multiplying the input, x, where x=the text features
        self.W_f = self.add_weight(shape=(input_shape[-1],self.output_size),name='W_f',initializer='glorot_uniform')
        self.W_i = self.add_weight(shape=(input_shape[-1],self.output_size),name='W_i',initializer='glorot_uniform')
        self.W_s = self.add_weight(shape=(input_shape[-1],self.output_size),name='W_s',initializer='glorot_uniform')
        self.W_c = self.add_weight(shape=(input_shape[-1],self.output_size),name='W_c',initializer='glorot_uniform')
        
        # Weights (matrix) multiplying the recursive cell output, output_prev
        self.U_f = self.add_weight(shape=(self.output_size,self.output_size),name='U_f',initializer='glorot_uniform')
        self.U_i = self.add_weight(shape=(self.output_size,self.output_size),name='U_i',initializer='glorot_uniform')
        self.U_s = self.add_weight(shape=(self.output_size,self.output_size),name='U_s',initializer='glorot_uniform')
        self.U_c = self.add_weight(shape=(self.output_size,self.output_size),name='U_c',initializer='glorot_uniform')
        
        # Bias weights (vector)
        # For b_f, the bias for the forget gate (where 0=forget), starts at 1 so that default is to not forget. 
        self.b_f = self.add_weight(shape=(self.output_size,),name='b_f',initializer='ones')
        self.b_i = self.add_weight(shape=(self.output_size,),name='b_i',initializer='zeros')
        self.b_s = self.add_weight(shape=(self.output_size,),name='b_s',initializer='zeros')
        self.b_c = self.add_weight(shape=(self.output_size,),name='b_c',initializer='zeros')
        
        self.built = True

    def call(self, inputs, states):
        cell_prev = states[0]
        output_prev = states[1]
        
        
        #f forget    = sigma(W_f*x  +  U_f*ouput_prev  +  b_f)    :decides how much of cell_state to keep/forget
        f = tf.keras.backend.dot(inputs, self.W_f) + tf.keras.backend.dot(output_prev, self.U_f) + self.b_f
        f = tf.keras.backend.sigmoid(f)
        
        #i input     = sigma(W_i*x  +  U_i*ouput_prev  +  b_i)    :decides which new candidate values to remember
        i = tf.keras.backend.dot(inputs, self.W_i) + tf.keras.backend.dot(output_prev, self.U_i) + self.b_i
        i = tf.keras.backend.sigmoid(i)
        
        #s select    = sigma(W_s*x  +  U_s*ouput_prev  +  b_s)    :decides what portions of the cell_state to output
        s = tf.keras.backend.dot(inputs, self.W_s) + tf.keras.backend.dot(output_prev, self.U_s) + self.b_s
        s = tf.keras.backend.sigmoid(s)
        
        #c candidate =  tanh(W_c*x  +  U_c*ouput_prev  +  b_c)    :creates new candidate values to be added to cell_state
        c = tf.keras.backend.dot(inputs, self.W_c) + tf.keras.backend.dot(output_prev, self.U_c) + self.b_c
        c = tf.keras.backend.tanh(c)
        
        #cell_state = (forget*cell_prev)+(input*candidate)
        #element-wise multiplication and addition 
        cell_state = tf.add( tf.multiply(f, cell_prev), tf.multiply(i, c) )
        
        #output = select*tanh(cell_state)
        output = tf.multiply(s, tf.keras.backend.tanh(cell_state))

        state = (cell_state, output)
        return output, state
    
    

#
# Merge architecture Neural Network.
#


# Define the layers of the neural network. Assuming EfficiencyNetB0 (ENB0) as the image CNN
def mergeModel(vocab_size, max_num_words):
  # vocab_size is the number of unique words in the set of training captions.
  # max_num_words is the length of the longest wordcount caption in the training set.

  # The merge model can also be described as an encoder-decoder model

  #
  # Encoders
  #

  # Encode image features from CNN ENB0's output to shape (128).
  # The https://arxiv.org/abs/1708.02043 paper suggests 128 is optimal for flikr30k out of 128, 256, 512.
  # Regadless of optimal accuracy, 128 will have fewer parameters and be faster to train, so we prefer it.
  img_input_pooled_features = tf.keras.layers.Input(shape=(1280))
  # Dropout layers discard some of the weights to help prevent overfitting the training data.
  img_avg_pool_regularized = tf.keras.layers.Dropout(0.25)(img_input_pooled_features) # shape (1280)
  img_128_features = tf.keras.layers.Dense(128, activation='relu')(img_avg_pool_regularized) # shape (128)

  # Encode linguistic/text sequence features.
  # Reccurrent Neural Network (RNN) works well for sequences.
  # LSTM as the RNN to give improved long term memory, i.e. try to not forget how the sentence started.
  txt_input = tf.keras.layers.Input(shape=(max_num_words,)) # shape (max_num_words)
  # An embedding layer converts an input word to a one-hot vector of length vocab_size and is then
  # densely (fully) connected to a desired output feature vector length (128).
  # This is similar in function, after training, to a word vectorizor like word2vec.
  txt_embeding = tf.keras.layers.Embedding(vocab_size, 128, mask_zero=True)(txt_input) # shape (max_num_words, 128)
  # Dropout for less overfitting (still there was quite a bit of overfitting, we did not try other values for dropout%, but would next time)
  txt_embeding_regularized = tf.keras.layers.Dropout(0.35)(txt_embeding) # shape (max_num_words, 128)
  # LSTM (with above implemented LSTM_Cell class)
  txt_128_features = tf.keras.layers.RNN(LSTM_Cell(128))(txt_embeding_regularized) # shape (128)

  
  # 
  # Decoder
  #

  # Merge the two encoder models with addition as the merging function.
  merger = tf.keras.layers.add([img_128_features, txt_128_features]) # shape (128)
  # Decode features with fully connected layer
  decoder = tf.keras.layers.Dense(128, activation='relu')(merger) # shape (128)
  # Output probability distribution over the vocabulary using softmax function.
  # The highest probability word becomes selected as the next word in the sequence.
  next_word = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder) # shape (vocab_size)


  #
  # Print and return the full model
  #

  model = tf.keras.Model(inputs=[img_input_pooled_features, txt_input], outputs=[next_word])
  print(model.summary())
  # Keras docs reccomend crossentropy loss and either rmsprop or adam optimizer
  model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

  return model

# COMMAND ----------

from keras.preprocessing.text import Tokenizer

# Adds a tokenizer to caption
def tokenize(captions):
  # Converts dictionary into nested list of captions
  captionLst = list(captions.values())
  # Flatmaps the captions so all captions can go into the tokenizer
  allCaptions = [item for sublist in captionLst for item in sublist]
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(allCaptions)
  return tokenizer

# COMMAND ----------

# Calculates the longest caption length
def calcMaxLength(captions):
  # Gets the nested list of captions
  captionLst = list(captions.values())
  # Flatmaps the captions
  captionListFlat = [item for sublist in captionLst for item in sublist]
  maxLength = 0
  # Iterates and inds the longest caption length
  for caption in captionListFlat:
    lengthCheck = len(caption.split())
    if lengthCheck > maxLength:
      maxLength = lengthCheck
  return maxLength

# Removes the longest captions
def removeLong(captions):
  # Gets the nested list of captions
  captionLst = list(captions.values())
  keysToRem = []
  # Iterates through each caption
  for key, caption in captions.items():
    for cap in caption:
      lengthCheck = len(cap.split())
      # Removes captions that are longer than 20 words
      if lengthCheck > 20:
        keysToRem.append(key)
  for keyRem in keysToRem:
    captions.pop(keyRem, None)
  return captions

# COMMAND ----------

from numpy import array
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Generates replicated caption data for the model to train on
def mapModelData(captions, photoFeats, tokenizer, maxLength, vocabSize):
  X1 = []
  X2 = [] 
  y = []
  # Iterates through each caption (5 per photoId)
  for photoId, caption in captions.items():
    partialCaption = []
    expectedNextWord = []
    for c in caption:
      seq = tokenizer.texts_to_sequences([c])[0]
      # For each caption, the model needs to learn how to predict each word individually.
      # An m-word caption is replicated into partial captions of length 1, 2,...,m-1 where the 
      # Expected output word is the next word in the original caption.
      for i in range(1, len(seq)):
        partialCaption = seq[:i]
        expectedNextWord = seq[i]
        # All caption inputs are padded to fill the full input vector.
        partialCaption = pad_sequences([partialCaption], maxlen=maxLength)[0]
        # word->integer embedding
        expectedNextWord = to_categorical([expectedNextWord], num_classes=vocabSize)[0]
        id = photoId.split(".")[0]
        X1.append(photoFeats[id]) 
        X2.append(partialCaption)
        y.append(expectedNextWord)
  return array(X1), array(X2), array(y)

# COMMAND ----------

import itertools
import pandas as pd
import os 

# load list of images (16K)
directory = "/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/flickr30k_images/flickr30k_images/"
imagesList = os.listdir(directory)

# loads captions
loadCaptions = pd.read_csv("/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/results.csv", delimiter='|')
loadCaptions.columns = ['image_name', 'comment_number', 'comment']

# photo features
#photoFeats = pd.read_pickle('/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/features.pkl') 
photoFeatsPooled = pd.read_pickle('/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/featuresPooled.pkl')

# Formats the captions and removes unecessarily long descriptions
captions = imagesMapCaption(imagesList, loadCaptions)
del captions["2199200615.jpg"] #this caption was giving an error. the entry in the dataset had a malformed last caption
formattedCaptions = formatCaptions(captions)
removeLong(formattedCaptions)

# Gets total # of features and # of features we want to train (80%)
totalCount = len(formattedCaptions)
trainCount = (int) (totalCount * .8)

# Gets 80% of the captions for training
trainCaptions = dict(itertools.islice(formattedCaptions.items(), trainCount))

# Gets 20% of the captions for testing
testCaptions = dict(itertools.islice(formattedCaptions.items(), trainCount, totalCount))

# Map w/ tokenizer and calculate size
tokenizer = tokenize(formattedCaptions)
vocabSize = len(tokenizer.word_index) + 1

# Calculate maximum length in a given segment
maxLength = calcMaxLength(formattedCaptions)

# COMMAND ----------

# maps our data for the training
X1train, X2train, ytrain = mapModelData(trainCaptions, photoFeatsPooled, tokenizer, maxLength, vocabSize)

# maps our data for the testing
X1test, X2test, ytest = mapModelData(testCaptions, photoFeatsPooled, tokenizer, maxLength, vocabSize)

# COMMAND ----------

from keras.callbacks import ModelCheckpoint

model = mergeModel(vocabSize, maxLength)

filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

model.fit([X1train, X2train], ytrain, epochs=10, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))

# COMMAND ----------

dbutils.fs.ls("file:///databricks/driver/")

# COMMAND ----------

# dbutils.fs.mv('dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/', '/FileStore/shared_uploads/axc200021@utdallas.edu/model-ep004-loss3.823-val_loss4.501.h5')
dbutils.fs.ls("/FileStore/shared_uploads/axc200021@utdallas.edu/")

# COMMAND ----------

from numpy import argmax
from nltk.translate.bleu_score import corpus_bleu

# Generate a description for an image
def imgCaption(model, photoFeatsPooled, maxLength, tokenizer):
  # Gets tokenizer's dictionary of word indexes
  tokenizerDict = tokenizer.word_index
  # Starts with beginning token
  textSeq = '<begofseq>'
  for i in range(maxLength):
    sequence = tokenizer.texts_to_sequences([textSeq])[0]

    sequence = pad_sequences([sequence], maxlen=maxLength)[0]

    input1 = tf.convert_to_tensor([photoFeatsPooled], dtype=tf.float32)
    input2 = tf.convert_to_tensor([sequence], dtype=tf.int64)
    
    # predict next word
    genInt = model.predict([input1, input2])
    
    # Gets word with highest probability
    genInt = argmax(genInt)
    
    if genInt in tokenizerDict:
        genWord = tokenizerDict[genInt]
    else:
        break
    # Ends caption if the end of it is reached
    if genWord == 'endofseq':
      textSeq += ' <' + genWord + '>'
      break
    textSeq += ' ' + genWord
  return textSeq

def modelEvaluation(model, captions, photoFeatsPooled, maxLength, tokenizer):
  original, predicted = [], []
  for key, caption in captions.items():
    # model generates description
    genCaption = imgCaption(model, photoFeatsPooled[key[:-4]], maxLength, tokenizer)
    references = [c.split() for c in caption]
    original.append(references)
    predicted.append(genCaption.split())
  print('BLEU-1: %f' % corpus_bleu(original, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(original, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(original, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(original, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# COMMAND ----------

# load new instance of the model for testing and evaluation
modelTest = mergeModel(vocabSize, maxLength)
filename = '/dbfs/FileStore/shared_uploads/axc200021@utdallas.edu/model-ep004-loss3.823-val_loss4.501.h5'
modelTest.load_weights(filename)

# COMMAND ----------

modelEvaluation(modelTest, testCaptions, photoFeatsPooled, tokenizer, maxLength)

# COMMAND ----------


captionsToS = dict(list(testCaptions.items())[:10])

for photoId, captions in captionsToS.items():
  yhat = imgCaption(modelTest, photoFeatsPooled[photoId[:-4]], maxLength, tokenizer)
  print("Key: ", photoId)
  print("Original: ", captions)
  print("Prediction: ", yhat)
  print('\n')

# COMMAND ----------


