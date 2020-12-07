# Final-Project-Group1
ML2 Final Project for Group 1

## Introduction and Project Summary
Our group was interested in exploring Recurrent Neural Networks (RNNs), so we chose a project focused on predicting the next value in a given sequence.  Specifically, we analyzed 3.6 million US airline flights over a period of 6 months.  Each record in our dataset includes the airline name, a unique identifier for each aircraft, the origin and destination airports, and the time of the flight.  We defined a flight as one unique aircraft flying from one of about 350 airports to another.  Our research question is “Given a sequence of N airports visited by a unique plane, can we predict the next airport (N+1)?”

We believe this approach is applicable to numerous different problems beyond predicting the next airport.  Our world is awash in devices that record their time and location, leading to an explosion of geospatial data that can inform everything from advertising to pandemic responses.  Often, the data is too dense for traditional geospatial analysis methods and a common approach is to represent a dataset as a network or sequence of known locations visited.  Applying a similar methodology as the one proposed will likely lead to additional insights in multiple fields and business cases.

## Code Summary
To facilitate efficient model development, training, and evaluation in an environment with multiple different versions of input datasets, we developed a data ingest and preprocessing pipeline using a series of Python scripts, which are detailed below.  This repo is intended to be cloned to the root directory of the user, and all directory paths are hard codes to look for the repo main folder at "~/".

To manage the different types of models and versions of our data, which can be segmented by airline and divided into different length sequences in the model pipeline, we used different “model_name” and “run_name” variables to track activity across different RNN models (LSTM, GRU, seq2seq, etc) and airline/sequence length “runs” respectively.  These variables for saving and loading data and models are found at the beginning of each script.

## Running this Code
The repo is configured to run a baseline model with a two-layer LSTM for Delta Airlines with a sequence length of 50.  If the repo is cloned at the root file '/home/ubuntu', a user must first run the EDA_and_cleaning.py script in '/home/ubuntu/Final-Project-Group1/Code/baseline_scripts' to install any needed Python packages and unzip the Data directory.  After that, a user can run any of the scripts in the '/home/ubuntu/Final-Project-Group1/Code/baseline_scripts'  in any order to explore a baseline model for Delta Airlines with a sequence length of 50.  Running additional models will require generation of some of those models and required data first.  Please see flow chart below for additional details on data processing.


![Data Processing Overview](https://github.com/redhairedcelt/Final-Project-Group1/blob/main/data_processing.png)

### EDA_and_cleaning.py:
- Read in csv data files that were downloaded from our source site as pandas dataframes.
- Concat all pandas dataframes into one df and sort by the unique id, date, and time to ensure temporal order
- Explore summary of feature counts and provide option to subset to a specific airline.
- Each row of the DataFrame lists a flight, with an origin, destination, unique aircraft ID, and additional information
- We then apply the build_history function, which uses multiprocessing to iterate across the large DataFrame.  The DataFrame is divided into chunks about 100,000 long, with all flights by a unique aircraft kept in the same DataFrame.  The result is a dictionary where the key is the unique aircraft and the value is a str with each each of the airports visited in order concatenated together.
- The separate pieces from the multiprocessing are then combined.  
- The resulting dict is written out as a DataFrame for the next step in the pipeline.  In addition to columns for the unique aircraft and the sequence of airports itself, we added a third column that counts the total length of the sequence.  

### ACF and seasonality.py
- Examines the autocorrelation of different subsets and the entire data by through generation of ACF plots.
- Test for stationarity (the absence of trend or seasonality) through Augmented Dickey-Fuller (ADF) test.
- Compare results to those of random sequences.

### preprocessing.py:
- Read in the DataFrame from the previous script. 
- This str is split into a numpy array, and then a broadcast function is applied that takes the numpy array, a window length, and a stride to generate an array of arrays each the length of the window.  So if there were originally 7 elements [0,1,2,3,4,5,6] and the window is 3 and stride is 1, we would get
[[0,1,2],
[1,2,3],
[2,3,4],
[3,4,5],
[4,5,6]]
- We do this for every unique aircraft and concatenate all the resulting sequences, which have a shape of n (number of series) by the window length.
- We then use the Keras tokenizer to translate the str elements into integers.
- Finally, we generate y by slicing off the last column of the resulting series of sequences and one-hot encoidng it using Keras "to_categorical" function.  This will be our target for one-step ahead prediction and is saved as y.npy.  The remaining data is now in the shape of n (number of series) by the window length - 1 and is saved as x.npy.

### modeling.py:
- Load x and y data as well as the tokenizer.
- Define run and model name variables.
- Set model hyperparameters.
- Build, compile, and fit model.
- Save off final model for later evaluation.

### modeling_seq2seq.py
- Same function as modeling.py, but tailored for unique architecture of seq2seq models.

### model_evaluate.py:
- Required x and y data, models, corresponding history files and tokenziers are loaded from disk. 
- Plot each model’s training history to compare the training and validation loss to ascertain if the model overfit or underfit.  
- Determine the overall accuracy and Cohen's Kappa score for each model.  
- Generate a confusion matrix and a classification report to evaluate where our models did well and where they did poorly.  
- Plot a scatterplot of the model's precision and recall for target predictions (next airport visited).  
