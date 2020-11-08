import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load
import datetime
import os

from sklearn.model_selection import train_test_split

from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding

# %%

run_name = 'DL_25'
print(f'Run name is {run_name}.')
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())


# load tokenizer, get vocab_size, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
x = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x.npy')
y = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y.npy')

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
y_train, y_test = to_categorical(y_train, num_classes=vocab_size), to_categorical(y_test, num_classes=vocab_size)
X_seq_length = x.shape[1]

# hyperparameters
N_NEURONS = 100
N_EPOCHS = 10
BATCH_SIZE = 512
EMBEDDING_SIZE = 300
DROPOUT = 0.2

# define model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
model.add(LSTM(N_NEURONS, return_sequences=True))
model.add(LSTM(N_NEURONS))
model.add(Dropout(DROPOUT))
model.add(Dense(N_NEURONS, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
print(model.summary())
# %%
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                    validation_data=(x_test, y_test), verbose=2)
accuracy = round((100 * model.evaluate(x_test, y_test)[1]), 3)
print(f"Final accuracy on validations set: {accuracy}")

# %%
# Visualize training process
plt.plot(history.history['loss'], label='Categorical crossentropy loss (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy loss (validation data)')
plt.title(f'Categorical crossentropy loss for {run_name}, overall accuracy: {accuracy}')
plt.ylabel('Categorical crossentropy loss value')
plt.yscale('log')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
# %%
# save the model to file
model.save(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_model.h5')
# save history
df_history = (pd.DataFrame.from_dict(history.history)
              .to_csv(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_history.csv'))