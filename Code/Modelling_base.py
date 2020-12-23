import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load
import datetime
import os

from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GlobalMaxPooling1D

# %%
learn_rate = [1e-3]
run_name = 'test_100'

path = os.getcwd()
# load tokenizer, get vocab_szie, and load x, y
tokenizer = load(open(path+f'/{run_name}_tokenizer_AA.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
#Generated Random sequence is loaded
x = np.load(path+f'/{run_name}_AA_x.npy')
y = np.load(path+f'/{run_name}_AA_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
y_train, y_test = to_categorical(y_train, num_classes=vocab_size), to_categorical(y_test, num_classes=vocab_size)
X_seq_length = x.shape[1]


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
y_train, y_test = to_categorical(y_train, num_classes=vocab_size), to_categorical(y_test, num_classes=vocab_size)


# hyperparameters
N_NEURONS = 200
N_EPOCHS = 20
BATCH_SIZE = 512
EMBEDDING_SIZE = 500
DROPOUT = 0.2


#LSTM
# define model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(LSTM(N_NEURONS))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)


#Stacked LSTM
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(LSTM(N_NEURONS,return_sequences=True))
model.add(LSTM(N_NEURONS-100))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)


#Bidirectional LSTM
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(Bidirectional(LSTM(N_NEURONS, activation='relu')))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)



#Conv1d
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length, output_dim=(150,1)))
model.add(TimeDistributed(Conv1D(64, kernel_size=1, activation='relu')))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(N_NEURONS-100))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)


#Conv1d
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(Conv1D(EMBEDDING_SIZE, 8, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(EMBEDDING_SIZE, 8, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)

# GRU
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(GRU(X_seq_length))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)

#GRU with Dropout
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(GRU(X_seq_length, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)

# Stacked GRU Network
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(GRU(X_seq_length, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(GRU(X_seq_length, activation='relu', dropout=0.1, recurrent_dropout=0.5))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)


#Conv1d with GRU - epoch 9  converges - 0.37
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(Conv1D(EMBEDDING_SIZE, 8, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(EMBEDDING_SIZE, 8, activation='relu'))
model.add(GRU(X_seq_length, dropout=0.1, recurrent_dropout=0.5))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)








