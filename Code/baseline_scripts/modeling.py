import numpy as np
import pandas as pd
from pickle import load

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding

# %%
# set run name
run_name = 'DL_50'
# set model name
model_name = 'baseline'
print(f'Run name is {run_name} and model name is {model_name}.')

# load tokenizer, get vocab_size, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
x_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_train.npy')
y_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy')

X_seq_length = x_train.shape[1]

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
# save the model to file
model.save(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
# save history
df_history = pd.DataFrame.from_dict(history.history)
df_history.to_csv(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_history.csv')
