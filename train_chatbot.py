# training chatbot
#* importing libraries
import numpy as np
from keras.models import Sequential # Sequential model
from keras.layers import Dense, Activation, Dropout # Dense layer, Activation function, Dropout layer
from keras.optimizers import SGD # Stochastic Gradient Descent
import random

import nltk # Natural Language Toolkit
from nltk.stem import WordNetLemmatizer # Lemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

#* defining intent file
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

#* pre-processing data
# tokenizing words: turning sentences into words
words = []
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenizing each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # add documents in the corpus
        documents.append((word, intent['tag']))
        # add to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)