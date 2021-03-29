import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from keras_preprocessing.text import Tokenizer
import pandas as pd

airline = 'DL'

#%%
def cal_acf(y_array, k):
    """
    Given a y series, finds the autocorrelations for given value k.
    Always 1 at lag 0.  Determines how much the kth lag of the dataset y_array
    correlates with the unlagged dataset.

    Args:
        y_array (list, np.array, iterable): time series or similar
        k (int): number of autocorrelations

    Returns:
        list: autocorrelations out to given lag k
    """
    result = []
    y_array = np.array(y_array)
    for lag in range(k):
        if lag == 0:
            r = 1
            result.append(r)
        else:
            yt = y_array[lag:]
            yt_minus_k = y_array[:-lag]
            r = ((np.sum((yt - y_array.mean()) * (yt_minus_k - y_array.mean()))) /
                 (np.sum((y_array - y_array.mean()) ** 2)))
            r = round(r, 5)
            result.append(r)
    return result


def plot_acf_data(acf_out, title=''):
    """
    Plots the ACF output of related function cal_acf as a symmetric function.
    Args:
        acf_out (list): ACF values for a dataset, the output of cal_acf

    Returns:
        plot: a symmetircal plot of the acf values
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stem(range(len(acf_out)), acf_out, use_line_collection=True)
    ax.stem(range(0, (-1 * len(acf_out)), -1), acf_out, use_line_collection=True)
    ax.set_title('{} ACF with k={}'.format(title, len(acf_out)))
    plt.show()


def adf_test(series, name, alpha=.05, rounded_to=3):
    """
    Parameters
    ----------
    series : A Pandas series
        A list of values to preform ADF on.
    name : str
        name for the print out report.
    rounded_to : int
        Number of decimals to round to.  default is 5.

    Returns
    -------
    A print out report of the Augmented Dickey-Fuller test.

    """
    adf_results = adfuller(series.copy().dropna())  # dropna if None or nan present
    adf_p = round(adf_results[1], rounded_to)
    test_stat = round(adf_results[0], rounded_to)
    print('\n')
    print('The ADF test for {} yields a p-value of {}.'.format(name, adf_p))
    if adf_p >= alpha:
        print("The p-value is equal to or greate than alpha.")
        print("This suggests the dataset is non-stationary.")
    if adf_p < alpha:
        print("The p-value is less than alpha.")
        print("This suggests the dataset is stationary.")

    print()
    print('The test statistic is {} and the critical values are:'.format(test_stat))
    for k, v in adf_results[4].items():
        print(k, round(v, rounded_to))

def sampen(L, m):
    """

    """
    N = len(L)
    r = (np.std(L) * .2)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([L[i: i + m] for i in range(N - m)])
    xmj = np.array([L[i: i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i: i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)
#%%
df_history = pd.read_csv(f'Data/flight_number_{airline}_sequence_hist.csv')

#%% Check Autocorrelation and Stationarity for the entire dataset
seq = df_history.seq.iloc[0]
seq_array = np.array(seq.split(' '))
all_sequences = seq_array.copy()

for i in range(1, len(df_history)):
    seq = df_history.seq.iloc[i]
    seq_array = np.array(seq.split(' '))
    all_sequences = np.concatenate([all_sequences, seq_array])

# use Keras' tokenizer to translate the str representations of airports into integers
# with a mapping kept in the tokenizer.  vocab_size will be a parameter for the network.
tokenizer = Tokenizer(lower=False, char_level=False)
# fit_on_texts builds the dict
tokenizer.fit_on_texts(all_sequences.tolist())
sequences_from_tokenizer = np.array(tokenizer.texts_to_sequences(all_sequences))

acf_x = cal_acf(sequences_from_tokenizer, k=100)
plot_acf_data(acf_x, title=f'ACF for all data with length {len(all_sequences)}')

# the adf test on all data is a heavy-compute process
#adf_test(pd.Series(sequences_from_tokenizer.reshape(-1)), name=f'ADF_Full')

#%% Check a random sequence for an aircraft
#i = 482
i = 744 # good lag out to 10 or so
#i = (np.random.randint(low=0, high=len(df_history), size=1))[0]
seq = df_history.seq.iloc[i]
seq_array = np.array(seq.split(' '))
# use Keras' tokenizer to translate the str representations of airports into integers
# with a mapping kept in the tokenizer.  vocab_size will be a parameter for the network.
tokenizer = Tokenizer(lower=False, char_level=False)
# fit_on_texts builds the dict
tokenizer.fit_on_texts(seq_array.tolist())
sequences_from_tokenizer = np.array(tokenizer.texts_to_sequences(seq_array))

acf_x = cal_acf(sequences_from_tokenizer, k=250)
plot_acf_data(acf_x, title=f'ACF for IDX {i} with length {len(seq_array)}')

adf_test(pd.Series(sequences_from_tokenizer.reshape(-1)), name=f'ADF Sample for Idx {i}')

#%% compare to random sequnece of airports of the same length
random_array = np.random.randint(low=1, high=len(tokenizer.word_index) + 1,
                              size=len(sequences_from_tokenizer))

acf_x = cal_acf(random_array, k=250)
plot_acf_data(acf_x, title=f'Random Array')

adf_test(pd.Series(random_array), name='Random Array')

#%% sample entropy
L_seq = sequences_from_tokenizer.flatten()
unique_vals = np.unique(L_seq)
sample_L_len = round(len(L_seq) / len(unique_vals))
rand_L_seq = np.random.choice(np.array(unique_vals), len(L_seq))

print(f'For sequence with length {len(L_seq)} with std of {np.std(L_seq)}...')
for i in range(1,10):
    print(f'For i = {i}:, true entropy is {round(sampen(L_seq, i),3)}')
    print(f'For i = {i}:, random entropy is {round(sampen(rand_L_seq, i), 3)}')


#%%
import collections

from scipy.stats import entropy


sequence = L_seq
bases = collections.Counter([tmp_base for tmp_base in sequence])
# define distribution
dist = [x / sum(bases.values()) for x in bases.values()]

# use scipy to calculate entropy
entropy_value = entropy(dist, base=2)
print((entropy_value))
#%% test for length 3
unique_vals = np.unique(L_seq)
sample_L_len = round(len(L_seq) / len(unique_vals))
rand_L_seq = np.random.choice(np.array(unique_vals), len(L_seq))


