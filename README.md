# Final-Project-Group1
ML2 Final Project for Group 1

EDA_and_cleaning.py:
-read in csv data files that were downloaded from our source site as pandas dataframes.
-concat all pandas dataframes into one df and sort by the unique id, date, and time to ensure temporal order
-Explore summary of feature counts and provide option to subset to a specific airline.
-Each row of the DataFrame lists a flight, with an origin, destination, unique aircraft ID, and additional information
-We then apply the build_history function, which uses multiprocessing to iterate across the large DataFrame.  The DataFrame is divided into chunks about 100,000 long, with all flights by a unique aircraft kept in the same DataFrame.  The result is a dictionary where the key is the unique aircraft and the value is a str with each each of the airports visited in order concatenated together.
-The separate pieces from the multiprocessing are then combined.  
-The resulting dict is written out as a DataFrame for the next step in the pipeline.  In addition to columns for the unique aircraft and the sequence of airports itself, we added a third column that counts the total length of the sequence.  You can see the csv for the entire dataset here.

preprocessing.py:
-We read in the DataFrame from the previous script, where each row includes the string of all airports visited by each aircraft.  
-This str is split into a numpy array, and then we apply a broadcast function that takes the numpy array, a window length, and a stride to generate an array of arrays each the length of the window.  So if there were originally 7 elements [0,1,2,3,4,5,6] and the window is 3 and stride is 1, we would get
[[0,1,2],
[1,2,3],
[2,3,4],
[3,4,5],
[4,5,6]]
-We do this for every unique aircraft and concatenate all the resulting sequences, which have a shape of n (number of series) by the window length.
-We then use the Keras tokenizer to translate the str elements into integers.
-Finally, we generate y by slicing off the last column of the resulting series of sequences.  This will be our target for one-step ahead prediction and is saved as y.npy.  The remaining data is now in the shape of n (number of series) by the window length - 1 and is saved as x.npy.

modeling.py:
This script is where we actually load our x and y arrays and start our different experiments with model architecture.
