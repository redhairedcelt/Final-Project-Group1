# %%
import pandas as pd
import numpy as np
import datetime
import pickle
from sklearn.model_selection import train_test_split
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical

# %%
# set the length of the sequence to extract from the data.
seq_len = 50
airline = 'DL'
# this is the size of the sliding window.
run_name = f'{airline}_{seq_len}'
print(f'Run name is {run_name}.')
df_history = pd.read_csv(f'/home/ubuntu/Final-Project-Group1/Data/flight_number_{airline}_sequence_hist.csv')


def broadcasting_app(a, l, s):  # Window len = l, Stride len/stepsize = s
    nrows = ((a.size - l) // s) + 1
    return a[s * np.arange(nrows)[:, None] + np.arange(l)]


first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

# Build the an array from the first row in the df so that dataset will be the right size to concat future rows with
# the rest of the rows
seq = df_history.seq.iloc[0]
seq_array = np.array(seq.split(' '))
first_seq = broadcasting_app(seq_array, l=seq_len, s=1)
all_sequences = first_seq.copy()

# iterate through the remaining rows of the df, get the sequence, broadcast to matrix, and concat
for i in range(1, len(df_history)):
    seq = df_history.seq.iloc[i]
    seq_array = np.array(seq.split(' '))
    seq_mat = broadcasting_app(seq_array, l=seq_len, s=1)
    all_sequences = np.concatenate([all_sequences, seq_mat])

# use Keras' tokenizer to translate the str representations of airports into integers
# with a mapping kept in the tokenizer.  vocab_size will be a parameter for the network.
tokenizer = Tokenizer(lower=False, char_level=False)
# fit_on_texts builds the dict
tokenizer.fit_on_texts(all_sequences.tolist())
# texts_to_sequences yields the integer valued sequences
sequences_from_tokenizer = np.array(tokenizer.texts_to_sequences(all_sequences.tolist()))
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# the features are the first n-1 airports, and the target is the last airport
x, y = sequences_from_tokenizer[:, :-1], sequences_from_tokenizer[:, -1:]

# split the dataset into test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
y_train, y_test = to_categorical(y_train, num_classes=vocab_size), to_categorical(y_test, num_classes=vocab_size)

# save off the data
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_train.npy', x_train)
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy', y_train)
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy', x_test)
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy', y_test)
# save the tokenizer
pickle.dump(tokenizer, open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL)

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)
