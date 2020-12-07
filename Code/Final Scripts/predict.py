# Loading necessary packages
from keras.models import load_model


def predictions(x):
    seq_len = x.shape[1]
    if seq_len == 25:
        run_name = 'DL_'+str(seq_len)
        model_name = 'lstm_1_400N'
        model = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
        predicts = model.predict(x)
        predicts_val = predicts.argmax(axis=1)
        # Mapping the predicts val to the index of the one_hot encoding to get the destination airport name
        # predicts_val
    elif seq_len == 50:
        run_name = 'DL_' + str(seq_len)
        model_name = 'lstm_1_300N'
        model = load_model(f'/home/ubuntu/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')
        predicts = model.predict(x)
        predicts_val = predicts.argmax(axis=1)
        # Mapping the predicts val to the index of the one_hot encoding to get the destination airport name
        # predicts_val
    return predicts_val


