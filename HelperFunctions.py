# -*- coding: utf-8 -*-

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np


"""
Functions to load previously processed image features and cleaned image descriptions
"""

# load doc into memory
def loadDoc(filename):
  # open the file as read only
  file = open(filename, 'r')
  # read all text
  text = file.read()
  # close the file
  file.close()
  return text

# load list of unique photo ids, derived from image file names
def loadImageIds(filename):
  doc = loadDoc(filename)
  dataset = list()
  # process line by line
  for line in doc.split('\n'):
    # skip empty lines
    if len(line) < 1:
      continue
    # get the image identifier
    identifier = line.split('.')[0]
    dataset.append(identifier)
  return set(dataset)

# load clean descriptions into memory
def loadCleanDescriptions(filename, dataset):
    # load document
    doc = loadDoc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
      # split line by white space
      tokens = line.split()
      # split id from description
      image_id, image_desc = tokens[0], tokens[1:]
      # skip images not in the set
      if image_id in dataset:
        # create list
        if image_id not in descriptions:
          descriptions[image_id] = list()
        # wrap description in tokens
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        # store
        descriptions[image_id].append(desc)
    return descriptions

# load photo features
def loadImageFeatures(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features


"""
Functions to process descriptions
"""

# Convert a dictionary of clean descriptions (image_id: list of descriptions) to a general list of all descriptions
def toLines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# map an integer to word
def wordForId(integer, tokenizer):
  for word, index in tokenizer.word_index.items():
    if index == integer:
      return word
  return none

# Calculate the length of the description with the most words
def calcMaxLength(description):
  lines = toLines(description)
  return max(len(d.split()) for d in lines)


"""
Functions to generate descriptions to caption new images
"""

# generate a description for an image using Glove 
def generateDesc(model, wordtoix, ixtoword, photo, max_length):
  # seed generation process with start flag
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence 
    sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    yhat = model.predict([photo, sequence], verbose=0)
    # convert probability to an integer
    yhat = argmax(yhat)
    # map integer to word
    word = ixtoword[yhat]
    # stop if we cannot map the word
    if word is None:
      break
    # append as input for generating the next word
    in_text += ' ' + word
    # stop if we predict the end of the sequence 
    if word == 'endseq':
      break
  return in_text