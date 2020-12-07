# Loading necessary packages
import pandas as pd
import numpy as np
import os
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, LSTM
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score, f1_score


def single_lstm(x_train,y_train,x_test,y_test,hyp_para,vocab_size, run_name,model_name):
    X_seq_length = x_train.shape[1]
    BATCH_SIZE = hyp_para[0]
    N_EPOCHS = hyp_para[1]
    N_NEURONS = hyp_para[2]
    EMBEDDING_SIZE = hyp_para[3]
    #############################################
    path = os.getcwd()
    #############################################
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
    model.add(LSTM(N_NEURONS))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # %%
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                        validation_data=(x_test, y_test), verbose=2,
                        callbacks=[ModelCheckpoint(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5',
                                                   monitor="val_loss", save_best_only=True)])
    model = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
    accuracy = round((100 * model.evaluate(x_test, y_test)[1]), 3)
    print(f"Final accuracy on validations set: {accuracy}")

    # save history
    df_history = pd.DataFrame.from_dict(history.history)
    df_history.to_csv(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_history.csv')

def stacked_lstm(x_train,y_train,x_test,y_test,hyp_para,vocab_size,run_name,model_name, stack_len):
    X_seq_length = x_train.shape[1]
    BATCH_SIZE = hyp_para[0]
    N_EPOCHS = hyp_para[1]
    N_NEURONS = hyp_para[2]
    EMBEDDING_SIZE = hyp_para[3]
    #############################################
    path = os.getcwd()
    #############################################
    model = Sequential()
    model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=X_seq_length))
    model.add(LSTM(N_NEURONS, return_sequences=True))
    for i in range(stack_len-2):
        model.add(LSTM(N_NEURONS - 100, return_sequences=True))
    model.add(LSTM(N_NEURONS - 100))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())

    # %%
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                        validation_data=(x_test, y_test), verbose=2,
                        callbacks=[ModelCheckpoint(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5',
                                                   monitor="val_loss", save_best_only=True)])
    model = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
    accuracy = round((100 * model.evaluate(x_test, y_test)[1]), 3)
    print(f"Final accuracy on validations set: {accuracy}")

    # save history
    df_history = pd.DataFrame.from_dict(history.history)
    df_history.to_csv(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_history.csv')

# Ensembling Technique
# Stacking Ensemble Model
# Reference - https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = np.dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX


# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # fit standalone model
    model = LogisticRegression()
    model.fit(stackedX, inputy)
    return model


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
    # create dataset using ensemble
    stackedX = stacked_dataset(members, inputX)
    # make a prediction
    yhat = model.predict(stackedX)
    return yhat


def ensemble_model(all_models,x_test,y_test,run_name,model_name):
    path = os.getcwd()
    # fit stacked model using the ensemble
    model = fit_stacked_model(all_models, x_test, np.argmax(y_test, axis=1))
    # evaluate model on test set
    yhat = stacked_prediction(all_models, model, x_test)
    # save the model to disk
    filename = run_name+"_"+model_name+"_model.sav"
    joblib.dump(model, filename)
    pred_y = yhat
    print("\n Final Best Model Results ")
    print("Final accuracy on validations set:", 100 * accuracy_score(pred_y, np.argmax(y_test, axis=1)), "%")
    print("Cohen Kappa", cohen_kappa_score(pred_y, np.argmax(y_test, axis=1)))
    print("F1 score", f1_score(pred_y, np.argmax(y_test, axis=1), average='macro'))



