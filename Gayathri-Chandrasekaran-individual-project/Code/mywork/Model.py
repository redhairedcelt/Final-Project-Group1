# Loading necessary packages
import numpy as np
from pickle import load
from Model_setup import single_lstm,stacked_lstm,ensemble_model
from Model_eval import eval
from keras.models import load_model

# Delta Airline - 25 Sequences
# set run name
run_name = 'DL_25'

# load tokenizer, get vocab_size, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
x_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_train.npy')
y_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy')
X_seq_length = x_train.shape[1]


# Model1 - Single Layer LSTM

# Hyper parameters Initialization
N_NEURONS = 400
N_EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING_SIZE = 200
model_name = 'lstm_1_400N'
hyp_para = [BATCH_SIZE, N_EPOCHS, N_NEURONS, EMBEDDING_SIZE]
# Model setup & Train the model
single_lstm(x_train, y_train, x_test, y_test, hyp_para, vocab_size, run_name, model_name)

#Model2 Stacked LSTM
model_name = 'lstm_3_400N'
stack_len = 3
# Model setup & Train the model
stacked_lstm(x_train, y_train, x_test, y_test, hyp_para, vocab_size, run_name, model_name, stack_len)

# Delta Airline - 50 Sequences
# set run name
run_name = 'DL_50'

# load tokenizer, get vocab_size, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1
x_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_train.npy')
y_train = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy')
X_seq_length = x_train.shape[1]

# Model1 - Single Layer LSTM

# Hyper parameters Initialization
N_NEURONS = 300
N_EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING_SIZE = 200
model_name = 'lstm_1_300N'
hyp_para = [BATCH_SIZE, N_EPOCHS, N_NEURONS, EMBEDDING_SIZE]
# Model setup & Train the model
single_lstm(x_train, y_train, x_test, y_test, hyp_para, vocab_size, run_name, model_name)
# Model2 Stacked LSTM
model_name = 'lstm_2_400N'
N_NEURONS = 400
stack_len = 2
hyp_para = [BATCH_SIZE, N_EPOCHS, N_NEURONS, EMBEDDING_SIZE]
# Model setup & Train the model
stacked_lstm(x_train, y_train, x_test, y_test, hyp_para, vocab_size, run_name, model_name, stack_len)

# Model Evaluation

# DL_25 Records
# Model 1
run_name = 'DL_25'
model_name = 'lstm_1_400N'
eval(run_name, model_name)
# Model 2
model_name = 'lstm_3_400N'
eval(run_name, model_name)

# DL_50 Records
# Model 1
run_name = 'DL_50'
model_name = 'lstm_1_300N'
eval(run_name, model_name)

# Model 2
model_name = 'lstm_2_400N'
eval(run_name, model_name)


# Ensemble Model
# DL_25 sequences
run_name = 'DL_25'
model_name = 'lstm_1_400N'
# load models
model1 = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
model_name = 'lstm_3_400N'
model2 = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy')
all_models = [model1, model2]
model_name = 'ensemble'
ensemble_model(all_models, x_test, y_test, run_name, model_name)

# DL_50 sequences
run_name = 'DL_50'
model_name = 'lstm_1_300N'
model3 = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
model_name = 'lstm_2_400N'
model4 = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
x_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_test.npy')
y_test = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_test.npy')
# DL_50 sequences
all_models = [model3, model4]
model_name = 'ensemble'
ensemble_model(all_models, x_test, y_test, run_name, model_name)