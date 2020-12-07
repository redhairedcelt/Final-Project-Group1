import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from pickle import load

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

from keras.models import Model
from keras.layers import Dense, Input
from keras.layers import LSTM

# set run name
run_name = 'DL_50'
# set model name
model_name = 'seq2seq'
print(f'Run name is {run_name} and model name is {model_name}.')

# load tokenizer, get vocab_size, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
x_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data 2/{run_name}_x_train.npy')
y_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data 2/{run_name}_y_train.npy')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data 2/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data 2/{run_name}_y_test.npy')

# Undo one hot encoding on target variables
y_train = np.argmax(y_train, axis=1)
y_train = np.reshape(y_train, (len(y_train), -1))
y_test = np.argmax(y_test, axis=1)
y_test = np.reshape(y_test, (len(y_test), -1))

# Combine x and y to be split however works best
x_train = np.concatenate((x_train, y_train), axis=1)
# Do the same for test data
x_test = np.concatenate((x_test, y_test), axis=1)

del y_train, y_test # No longer needed

# Check shapes before splitting
print(x_train.shape)
print(x_test.shape)
print()

# Define length of beginning sequence
split = 49

# Train a model for each split
# Results and models are saved to /home/ubuntu/Final-Project-Group1/Models/DL_25_seq2seq
seq_beg, other, seq_end = np.split(x_train, [split, split], axis=1) # Split data and analyze shapes
print(other.shape)
del other # remove this useless output
print(seq_beg.shape)
print(seq_end.shape)
print()

# Add special beginning and end tags to training ending sequences
seq_end = np.insert(seq_end, 0, 1111, axis=1)
seq_end = np.insert(seq_end, seq_end.shape[1], 9999, axis=1)
print(seq_end.shape)
print()

# Also split test data and analyze shapes
seq_beg_test, other, seq_end_test = np.split(x_test, [split, split], axis=1)
print(other.shape)
del other
print(seq_beg_test.shape)
print(seq_end_test.shape)
print()

# Add special beginning and end tags to testing ending sequences
seq_end_test = np.insert(seq_end_test, 0, 1111, axis=1)
seq_end_test = np.insert(seq_end_test, seq_end_test.shape[1], 9999, axis=1)

print(seq_end_test.shape)
print()

# Store all unique airport IDs in a list
airports = x_train.flatten().tolist()
airports.append(1111) # Add the special characters so they will be in dictionaries
airports.append(9999)
airports = set(airports)
airports = sorted(list(airports))

# dictionary to index each airport - key is index and value is airport
index_to_airport_dict = {}
# dictionary to get airport given its index - key is airport and value is index
airport_to_index_dict = {}
for k, v in enumerate(airports):
    index_to_airport_dict[k] = v
    airport_to_index_dict[v] = k

# Get empty numpy arrays to tokenize the training sequences
tokenized_seq_beg = np.zeros(shape=(seq_beg.shape[0], seq_beg.shape[1], len(airports)), dtype='float32')
tokenized_seq_end = np.zeros(shape=(seq_end.shape[0], seq_end.shape[1], len(airports)), dtype='float32')
target_data = np.zeros(shape=(seq_end.shape[0], seq_end.shape[1], len(airports)), dtype='float32')

# Vectorize the beginning and ending sequences for training data
for i in range(seq_beg.shape[0]):
    for k, ch in enumerate(seq_beg[i]):
        tokenized_seq_beg[i, k, airport_to_index_dict[ch]] = 1
    for k, ch in enumerate(seq_end[i]):
        tokenized_seq_end[i, k, airport_to_index_dict[ch]] = 1
        # decoder_target_data will be ahead by one timestep and will not include the start airport.
        if k > 0:
            target_data[i, k - 1, airport_to_index_dict[ch]] = 1

# Get empty numpy array to tokenize the beginning test sequences to be fed at evaluation time
tokenized_seq_beg_test = np.zeros(shape=(seq_beg_test.shape[0], seq_beg_test.shape[1], len(airports)), dtype='float32')

# Vectorize the beginning sequences for test data to be fed to encoder
for i in range(seq_beg_test.shape[0]):
    for k, ch in enumerate(seq_beg_test[i]):
        tokenized_seq_beg_test[i, k, airport_to_index_dict[ch]] = 1

# hyperparameters
N_NEURONS = 256
N_EPOCHS = 6
BATCH_SIZE = 64

# Encoder Model
encoder_input = Input(shape=(None, len(airports)))
encoder_LSTM = LSTM(N_NEURONS, return_state=True)
encoder_outputs, encoder_h, encoder_c = encoder_LSTM(encoder_input)
encoder_states = [encoder_h, encoder_c] # These states are passed to decoder LSTM layer

# Decoder model
decoder_input = Input(shape=(None, len(airports)))
decoder_LSTM = LSTM(N_NEURONS, return_sequences=True, return_state=True)
decoder_out, _, _ = decoder_LSTM(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(len(airports), activation='softmax')
decoder_out = decoder_dense(decoder_out)

model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_out])

# Run training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=[tokenized_seq_beg, tokenized_seq_end], y=target_data,
          batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=.2)

# Visualize training process
plt.plot(history.history['loss'], label='Categorical crossentropy loss (training data)')
plt.plot(history.history['val_loss'], label='Categorical crossentropy loss (validation data)')
plt.title(f'Categorical crossentropy loss for {run_name}, overall accuracy: {run_name}')
plt.ylabel('Categorical crossentropy loss value')
plt.yscale('log')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# save the model to file
model.save(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
# save history
df_history = pd.DataFrame.from_dict(history.history)
df_history.to_csv(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_history.csv')

######################################################################################################################

# Evaluate model with test data
# Encoder inference model
encoder_model_inf = Model(encoder_input, encoder_states)

# Decoder inference model
decoder_state_input_h = Input(shape=(N_NEURONS,))
decoder_state_input_c = Input(shape=(N_NEURONS,))
decoder_input_states = [decoder_state_input_h, decoder_state_input_c]
decoder_out, decoder_h, decoder_c = decoder_LSTM(decoder_input, initial_state=decoder_input_states)
decoder_states = [decoder_h, decoder_c]
decoder_out = decoder_dense(decoder_out)

decoder_model_inf = Model(inputs=[decoder_input] + decoder_input_states,
                          outputs=[decoder_out] + decoder_states)

def decode_seq(inp_seq):

    # Get initial states by feeding beginning of a test sequence to encoder
    states_val = encoder_model_inf.predict(inp_seq)

    # Set first target sequence to be 1111 (one hot encoded)
    decoder_input = np.zeros((1, 1, len(airports)))
    decoder_input[0, 0, airport_to_index_dict[1111]] = 1

    # Start predicted airplane route with special character
    airplane_route = [1111]
    stop_condition = False

    # Predict the next airports
    while not stop_condition:
        decoder_out, decoder_h, decoder_c = decoder_model_inf.predict(x=[decoder_input] + states_val)
        # Get index of predicted airport
        max_val_index = np.argmax(decoder_out[0, -1, :])
        sampled_airport = index_to_airport_dict[max_val_index]
        # Append predicted airport to list representing predicted sequence
        airplane_route.append(sampled_airport)

        # If predictions surpass the ending sequence length or model predicts 9999 indicating end of sequence
        if (sampled_airport == 9999) or (len(airplane_route) > (seq_end.shape[1] - 1)):
            stop_condition = True

        # Update predicted airport to be fed to decoder model
        decoder_input = np.zeros((1, 1, len(airports)))
        decoder_input[0, 0, max_val_index] = 1

        # Get states for predicting next character
        states_val = [decoder_h, decoder_c]

    return airplane_route # Return predicted sequence

cumulative_predictions = [] # To accumulate all predictions
cumulative_actuals = [] # To accumulate all actual labels
cumulative_accuracy = 0
test_sequences = 5000
drops = [1111, 9999] # specify beg and end tags to drop for evaluation
# Loop through test data and feed input sequences to encoder model
loop_count = 0
print('Beginning inference...')
for seq_index in range(test_sequences):
    inp_seq = tokenized_seq_beg_test[seq_index]
    inp_seq = np.expand_dims(inp_seq, axis=0) # Resize to go into encoder model
    pred_airplane_route = decode_seq(inp_seq)

    # Drop beginning and end tags before calculating evaluation metrics
    pred_airplane_route = [_ for _ in pred_airplane_route if _ not in drops]
    actual_airplane_route = seq_end_test[seq_index]
    actual_airplane_route = [_ for _ in actual_airplane_route if _ not in drops]

    # print('-')
    # print('Input sequence:', seq_beg_test[seq_index])
    # print('Predicted output sequence:', pred_airplane_route)
    # print('Actual output sequence:', actual_airplane_route)
    # print('Actual whole sequence', x_test[seq_index])

    correct, incorrect = 0, 0  # To keep track of right and wrong predictions
    for _ in range(len(actual_airplane_route)):
        if pred_airplane_route[_] == actual_airplane_route[_]:
            correct += 1
        else:
            incorrect += 1

        # Append predictions and labels to huge lists for classification report
        cumulative_predictions.append(pred_airplane_route[_])
        cumulative_actuals.append(actual_airplane_route[_])

    accuracy = correct / (correct + incorrect)
    #print('Test Accuracy', accuracy) # This gives the accuracy on each test sequence

    cumulative_accuracy += accuracy # Accumulate accuracy from all test sequences to be averaged later

    #loop_count += 1
    #print('Processing test sequence ' + str(loop_count) + ' out of ' + str(test_sequences))

######################################################################################################################

# Evaluate model performance on test data
cumulative_accuracy = cumulative_accuracy / test_sequences # Gets accuracy over all test sequences used

print()

# Get classification report
class_report = classification_report(cumulative_actuals, cumulative_predictions, output_dict=True)
print(class_report)
print()

# Get confusion matrix
conf_mat = confusion_matrix(y_true=cumulative_actuals, y_pred=cumulative_predictions)
print(conf_mat)
print()

# Get accuracy
print('Accuracy:', cumulative_accuracy)

# Get Cohens Kappa
ck_score = cohen_kappa_score(cumulative_actuals, cumulative_predictions)
print('Cohens Kappa:', ck_score)

# Get precision
print('Precision:', class_report['weighted avg']['precision'])

# Get recall
print('Recall:', class_report['weighted avg']['recall'])

# Get F1
print('F1:', class_report['weighted avg']['f1-score'])

# Get support
print('Support:', class_report['weighted avg']['support'])

# Create dataframe from classification report
df_class_report = pd.DataFrame(class_report).T.iloc[:-3,:]
df_class_report.sort_values('f1-score', inplace=True)
print(df_class_report)

# Plot the classes (airports) as a scatterplot colored by F1 and sized by total numbed of flights from each airport.
# g = sns.scatterplot(x='precision', y='recall', size='support',
#                     hue='f1-score', data=df_class_report)
# plt.title("Scatterplot of Model's Precision and Recall, \nColored by F1 Score, Sized by Number of Flights")
# plt.show()

plt.scatter(df_class_report['precision'], df_class_report['recall'], s=df_class_report['support'],
           c=df_class_report['f1-score'])
plt.title(f"Scatterplot of {model_name}_{run_name} Precision and Recall, \nColored by F1 Score, Sized by Number of Flights")
plt.show()

# proc log
log_name = f'/home/ubuntu/Final-Project-Group1/Logs/{model_name}'
log = open(log_name, 'a+')
log.write(f'{model_name} for {run_name} scored {accuracy} accuracy and {ck_score} cohen_kappa score. \n')
log.close()
