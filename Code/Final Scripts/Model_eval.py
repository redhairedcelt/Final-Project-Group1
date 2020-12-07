import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pickle import load
import os
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model



def eval(run_name,model_name):
    print(f'Run name is {run_name} and model name is {model_name}.')
    # load tokenizer, get vocab_size, and load x, y
    tokenizer = load(open(f'~/Final-Project-Group1/Models/{run_name}_tokenizer.pkl', 'rb'))
    x_test = np.load(f'~/Final-Project-Group1/Data/{run_name}_x_test.npy')
    y_test = np.load(f'~/Final-Project-Group1/Data/{run_name}_y_test.npy')
    # load history
    history = pd.read_csv(f'~/Final-Project-Group1/Models/{run_name}_{model_name}_history.csv')
    
    # load model
    model = load_model(f'~/Final-Project-Group1/Models/{run_name}_{model_name}_model.h5')

    # %% Get Accuracy and Plot the training/validation curves
    accuracy = round((100 * model.evaluate(x_test, y_test)[1]), 3)
    print(f'Run name is {run_name} and model name is {model_name}.')
    print("\nModal Evalution Results")
    print("\nAccuracy: ",accuracy)

    # Visualize training process
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history['loss'], label='Categorical crossentropy loss (training data)')
    plt.plot(history['val_loss'], label='Categorical crossentropy loss (validation data)')
    plt.title(f'Categorical crossentropy loss for {model_name}_{run_name}, \n overall accuracy: {accuracy}')
    plt.ylabel('Categorical crossentropy loss value')
    plt.yscale('log')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    #%% predict the models output based on x test and generate confusion matrix
    prediction = model.predict(x_test)
    #%%
    from sklearn.metrics import cohen_kappa_score
    ck_score = cohen_kappa_score(y_test.argmax(axis=1), prediction.argmax(axis=1))
    print("Cohen kappa score", ck_score )
    #%%
    # print confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
    conf_mat = confusion_matrix(y_true=tokenizer.sequences_to_texts([y_test.argmax(axis=1)])[0].split(' '),
                                y_pred=tokenizer.sequences_to_texts([prediction.argmax(axis=1)])[0].split(' '),
                                labels=list(tokenizer.word_index.keys()))
    #%% use the models predicted outputs to compare to the y_test.
    # this is complicated by:
    # 1. The need to select the highest value prediction in the multilabel predictions (use of argmax(
    # 2. The need to use the tokenizer to transform the integer token back to the airport name.
    # The "sequences_to_texts" method yields a str, which must be split.
    # 3. The desire to save the output as a s csv and view as a df, which requires the output to be a dit.

    class_report = classification_report(y_true=tokenizer.sequences_to_texts([y_test.argmax(axis=1)])[0].split(' '),
                                         y_pred=tokenizer.sequences_to_texts([prediction.argmax(axis=1)])[0].split(' '),
                                         labels=list(tokenizer.word_index.keys()), output_dict=True)
    # Get precision
    print('Precision:', class_report['weighted avg']['precision'])
    # Get recall
    print('Recall:', class_report['weighted avg']['recall'])
    # Get F1
    print('F1:', class_report['weighted avg']['f1-score'])
    # Get support
    print('Support:', class_report['weighted avg']['support'])
    # we skip the last three lines, which are micro, macro and weighted F1 Scores.
    # These throw off sorting by other values.
    df_class_report = pd.DataFrame(class_report).T.iloc[:-3,:]
    df_class_report.sort_values('f1-score', inplace=True)
    #%% Plot the classes (airports) as a scatterplot colored by F1 and sized by total numbed of flights from each airport.
    # we will experiment with both seaborn and matplotlib.
    import matplotlib.pyplot as plt
    import seaborn as sns
    g = sns.scatterplot(x='precision', y='recall', size='support',
                        hue='f1-score', data=df_class_report)
    plt.title("Scatterplot of Model's Precision and Recall, \nColored by F1 Score, Sized by Number of Flights")
    #plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()
    # plot using matplotlib
    #plt.scatter(df_class_report['precision'], df_class_report['recall'], s=df_class_report['support']/20,
    #            c=df_class_report['f1-score'])
    # plt.title(f"Scatterplot of {model_name}_{run_name} Precision and Recall, \nColored by F1 Score, Sized by Number of Flights")
    # plt.show()
    #%%
    # proc log
    log_name = f'~/Final-Project-Group1/Logs/{model_name}'
    log = open(log_name, 'a+')
    log.write(f'{model_name} for {run_name} scored {accuracy} accuracy and {ck_score} cohen_kappa score. \n')
    log.close()