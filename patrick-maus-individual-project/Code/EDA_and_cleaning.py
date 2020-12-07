import os
import datetime
from multiprocessing import Pool, set_start_method

# try pip installing and loading additional modules
try:
    os.system("pip install numpy")
    os.system("pip install matplotlib")
    os.system("pip install pandas")
    os.system("pip install keras")
    os.system("pip install scikit-learn")
    os.system("pip install sklearn")
    os.system("pip install tensorflow")
    os.system("pip install seaborn")
    os.system("pip install statsmodels")

except Exception as e:
    print('Failed to install modules.')
    print(e)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import zipfile
try:
    with zipfile.ZipFile('/home/ubuntu/Final-Project-Group1/Data_dir.zip', 'r') as zip_ref:
        zip_ref.extractall('/home/ubuntu/Final-Project-Group1')
    print('Data dir unzipped.')
except:
    print('Error unzipping Data dir...')

# %% read in each csv file as a separate df
sep = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/sep_2019.csv')
oct = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/oct_2019.csv')
nov = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/nov_2019.csv')
dec = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/dec_2019.csv')
jan = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/jan_2020.csv')
feb = pd.read_csv('/home/ubuntu/Final-Project-Group1/Data/air_data/feb_2020.csv')

# put together the month pieces
df_flights = pd.concat([sep, oct, nov, dec, jan, feb])
del (sep, oct, nov, dec, jan, feb)
# %% there are several options to combine the features into a unique flight number.
# after experimenting with different combos, using the unique tail number provided
# the most stability in predictions over time.
df_flights['tail_flight_combo'] = df_flights['TAIL_NUM'] + '_' + df_flights['OP_CARRIER_FL_NUM'].astype('str')
df_flights['carrier_flight_combo'] = (df_flights['OP_UNIQUE_CARRIER'] + '_' +
                                      df_flights['OP_CARRIER_FL_NUM'].astype('str'))

# print summaries of total counts
print(f'Total unique tail_num rows: {len(df_flights.TAIL_NUM.unique())}')
print(f'Total unique flight_num rows: {len(df_flights.OP_CARRIER_FL_NUM.unique())}')
print(f'Total unique carrier ID rows: {len(df_flights.OP_UNIQUE_CARRIER.unique())}')
print(f'Total unique combined carrier ID and flight number rows: {len(df_flights.carrier_flight_combo.unique())}')
print(f'Total unique combined tail number and flight number rows: {len(df_flights.tail_flight_combo.unique())}')

print(df_flights.groupby('OP_UNIQUE_CARRIER').agg('count'))

# %%
airline = 'DL'
# subset to just one airline
df_flights = df_flights[df_flights['OP_UNIQUE_CARRIER'] == airline]

# Clean and sort df_flights into an edgelist that defines all the different sites visited
# by each uid.  These must be sorted in time order!!
target = 'TAIL_NUM'
# need to sort so rows are sequential
df_flights = df_flights.sort_values([target, 'FL_DATE', 'DEP_TIME'])
df_edgelist = pd.concat([df_flights['ORIGIN'], df_flights['DEST'], df_flights[target]], axis=1)
# Source and Target should be capitalized so the network can be read into Gephi.
df_edgelist.columns = ['Source', 'Target', 'uid']
df_edgelist.dropna(inplace=True)

print(f'The target feature for this run is {target}.')
print(f'The total number of rows is {len(df_edgelist)}.')
print(f"The total of unique UIDs is {len(df_edgelist['uid'].unique())}.")

# %% Group by the airport to get total number of flights per airport.  sort by largest, build bar graph
top_size = 12
(df_edgelist.groupby('Source').agg('count')['uid'].sort_values(ascending=False).head(top_size).plot(kind='bar'))
plt.title(f'Top {top_size} Airport by Number of Flights for {airline}')
plt.show()

# Group by the airport, get top 25 busiest airports, sum all others, and build pie chart
top = df_edgelist.groupby('Source').agg('count')['uid'].sort_values(ascending=False).head(top_size)

top_visits = top.sum()
other_visits = len(df_edgelist) - top_visits
top_and_other = top.append(pd.Series(other_visits, index=['OTHER']))

top_and_other.rename('Flights per Airport', inplace=True)
top_and_other.plot(kind='pie')
plt.title(f'Percent of Flight From Top {top_size} Airports for {airline}')
plt.show()


# %% build history dict
def build_history(df_edgelist):
    # build a history that includes all site visits per uid as a dict with the uid as
    # the key and the strings of each site visited as the values
    # make sure to replace ' ' with '_' in the strings so multi-word sites are one str
    history = dict()
    # get all unique uids
    uids = df_edgelist['uid'].unique()
    for uid in uids:
        uid_edgelist = df_edgelist[df_edgelist['uid'] == uid]
        uid_str = ''
        # add all the sources from the source column
        for s in uid_edgelist['Source'].values:
            uid_str = uid_str + ' ' + (s.replace(' ', '_'))
        # after adding all the sources, we still need to add the last target.
        # adding all the sources will provide the history of all but the n-1 port
        uid_str = uid_str + ' ' + (uid_edgelist['Target'].iloc[-1].replace(' ', '_'))
        # only add this history to the dict if the len of the value (# of ports) is >= 2
        if len(uid_str.split()) >= 2:
            history[uid] = uid_str.strip()
    return history


# %%
# need to split the edgelist to relatively equal pieces with complete uid histories
# this is not memory efficent as it replicates the data into pieces, but it will be
# faster for building the history.
df_max_len = 100000
numb_dfs = (len(df_edgelist) // df_max_len)
uids = np.array(df_edgelist['uid'].unique())
split_uids = np.array_split(uids, numb_dfs)
list_df = []
for split in split_uids:
    df_piece = df_edgelist[df_edgelist['uid'].isin(split)]
    list_df.append(df_piece)

# %% Execute the build history function using pooled workers.
# id number of cores and set workers.  use n-1 workers to keep from crashing machine.
first_tick = datetime.datetime.now()
print('Starting Processing at: ', first_tick.time())

cores = os.cpu_count()
workers = cores - 1
print(f'This machine has {cores} cores.  Will use {workers} for multiprocessing.')

if __name__ == "__main__":
    set_start_method('fork')
    with Pool(workers) as p:
        history_pieces = p.map(build_history, list_df[:5000])

last_tock = datetime.datetime.now()
lapse = last_tock - first_tick
print('Processing Done.  Total time elapsed: ', lapse)

# %%
# recombine pieces
history = dict()
for piece in history_pieces:
    for k, v in piece.items():
        history[k] = v

# %%
df_history = pd.DataFrame([history.keys(), history.values()]).T
df_history.columns = ['flight_numb', 'seq']
df_history['numb_in_seq'] = df_history['seq'].str.split().map(len)
df_history.to_csv(f'/home/ubuntu/Final-Project-Group1/Data/flight_number_{airline}_sequence_hist.csv', index=False)
# %%
df_history['numb_in_seq'].hist()
plt.title(f'Histogram of Flight Numbers and Numbers of Stops in Each Sequence for {airline}')
plt.show()
