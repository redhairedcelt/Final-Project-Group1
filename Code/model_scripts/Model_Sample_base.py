import numpy as np
from pickle import load
import random
import os
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix, classification_report
from Model_setup import single_lstm
import warnings
warnings.filterwarnings('ignore')

run_name = 'DL_50'
path = os.getcwd()

# load tokenizer, get vocab_szie, and load x, y
tokenizer = load(open(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
vocab_size = len(tokenizer.word_index) + 1

x = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_x_train.npy')
y = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy')

# Generating random samples similar to shape of the x_train
x_check =[]
for i in range(0,len(x)):
    val = random.sample(range(0,50), 49)
    x_check.append(val)
x_check = np.array(x_check)
np.save(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_dummy_samples.npy', x_check)

x = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_dummy_samples.npy')
y = np.load(f'/home/ubuntu/Final-Project-Group1/Data/{run_name}_y_train.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
X_seq_length = x.shape[1]


# Model1 - Single Layer LSTM
# Hyper parameters
N_NEURONS = 200
N_EPOCHS = 10
BATCH_SIZE = 256
EMBEDDING_SIZE = 200
model_name = 'lstm_sample'
hyp_para = [BATCH_SIZE, N_EPOCHS, N_NEURONS, EMBEDDING_SIZE]
single_lstm(x_train, y_train, x_test, y_test, hyp_para, vocab_size, run_name, model_name)

# Model Evaluation
model = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
prediction = model.predict(x_test)
accuracy = round((100 * model.evaluate(x_test, y_test)[1]), 3)
print(f"Final accuracy on validations set: {accuracy}")
ck_score = cohen_kappa_score(y_test.argmax(axis=1), prediction.argmax(axis=1))
print("Cohen kappa score", ck_score)
class_report = classification_report(y_true=tokenizer.sequences_to_texts([y_test.argmax(axis=1)])[0].split(' '),
                                     y_pred=tokenizer.sequences_to_texts([prediction.argmax(axis=1)])[0].split(' '),
                                     labels=list(tokenizer.word_index.keys()), output_dict=True)
# Get precision
print('Precision:', class_report['weighted avg']['precision'])
# Get recall
print('Recall:', class_report['weighted avg']['recall'])
# Get F1
print('F1:', class_report['weighted avg']['f1-score'])



