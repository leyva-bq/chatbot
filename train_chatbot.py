# training chatbot
#* importing libraries
import numpy as np
from keras.models import Sequential # Sequential model
from keras.layers import Dense, Activation, Dropout # Dense layer, Activation function, Dropout layer
from keras.optimizers import SGD # Stochastic Gradient Descent
import random

import nltk # Natural Language Toolkit
from nltk.stem import WordNetLemmatizer # lemmatizer
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

# lemmatizing each word
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

"""
# documents = combination between patterns and intents
print(len(documents), "documents")

# classes = intents
print(len(classes), "classes", classes)

# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)
"""

pickle.dump(words, open('words.pkl', 'wb')) # wb = write binary

#* create training and testing data
# create training data
training = []

# create empty array for the output
output_empty = [0] * len(classes)

# training set
for doc in documents:
    # initialize bag of words
    bag = []

    # list of tokenized words for pattern
    word_patterns = doc[0]

    # lemmatize each word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # create bag of words array w 1, if word is found in current pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # output is 0 for each tag & 1 for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle the features & make numpy array
random.shuffle(training)
training = np.array(training)

# create training and testing lists: x = patterns, y = intents
train_x = list(training[:,0])
train_y = list(training[:,1])