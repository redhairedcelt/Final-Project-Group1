#%%
import pandas as pd
import numpy as np
import datetime
from pickle import dump
from keras_preprocessing.text import Tokenizer

#%%
run_name = 'DL_25'
airline = 'DL'
print(f'Run name is {run_name}.')
df_history = pd.read_csv(f'~/Final-Project-Group1/Data/flight_number_{airline}_sequence_hist.csv')
# set the length of the sequence to extract from the data.
# this is the size of the sliding window.
seq_len = 25

def broadcasting_app(a, L, S ):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size-L)//S)+1
    return a[S*np.arange(nrows)[:, None] + np.arange(L)]

first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())


# Build the an array from the first row in the df so that dataset will be the right size to concat future rows with
# the rest of the rows
seq = df_history.seq.iloc[0]
seq_array = np.array(seq.split(' '))
first_seq = broadcasting_app(seq_array, L=seq_len, S=1)
all_sequences = first_seq.copy()

# iterate through the remaining rows of the df, get the sequence, broadcast to matrix, and concat
# TODO: add parallelization to this to speed up testing multiple sequence lengths.
for i in range(1, len(df_history)):
    seq = df_history.seq.iloc[i]
    seq_array = np.array(seq.split(' '))
    seq_mat = broadcasting_app(seq_array, L=seq_len, S=1)
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
x, y = sequences_from_tokenizer[:,:-1], sequences_from_tokenizer[:,-1:]

# save off the data
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x.npy', x)
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y.npy', y)
# save the tokenizer
dump(tokenizer, open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'wb'))

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)