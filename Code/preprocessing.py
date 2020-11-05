#%%
from keras.preprocessing import timeseries_dataset_from_array
from keras_preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

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
for i in range(1, len(df_history)):
    seq = df_history.seq.iloc[i]
    seq_array = np.array(seq.split(' '))
    seq_mat = broadcasting_app(seq_array, L=seq_len, S=1)
    all_sequences = np.concatenate([all_sequences, seq_mat])

tokenizer = Tokenizer(lower=False, char_level=False)
tokenizer.fit_on_texts(all_sequences.tolist())
sequences_from_tokenizer = tokenizer.texts_to_sequences(all_sequences.tolist())
sequences_matrix = tokenizer.sequences_to_matrix(sequences_from_tokenizer)
