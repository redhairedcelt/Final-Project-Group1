#%%
import pandas as pd
import numpy as np

from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
#%%
df_history = pd.read_csv('flight_number_sequence_hist.csv')

def broadcasting_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:,None] + np.arange(L)]


seq_len = 50
# Build the an array from the first row in the df so that dataset will be the right size to concat future rows with
# the rest of the rows
seq = df_history.seq.iloc[0]
seq_array = np.array(seq.split(' '))
first_seq = broadcasting_app(seq_array, L=seq_len, S=1)
all_sequences = first_seq.copy()

# iterate through the remaining rows of the df, get the sequence, broadcast to matrix, and concat
for i in range(1, len(df_history[:100])):
    seq = df_history.seq.iloc[i]
    seq_array = np.array(seq.split(' '))
    seq_mat = broadcasting_app(seq_array, L=seq_len, S=1)
    all_sequences = np.concatenate([all_sequences, seq_mat])

tokenizer = Tokenizer(lower=False, char_level=False)
tokenizer.fit_on_texts(all_sequences.tolist())
sequences_from_tokenizer = np.array(tokenizer.texts_to_sequences(all_sequences.tolist()))
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
#%%
X, y = sequences_from_tokenizer[:,:-1], sequences_from_tokenizer[:,-1:]
y = to_categorical(y, num_classes=vocab_size)
y = to_categorical(y, num_classes=vocab_size)
X_seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))
