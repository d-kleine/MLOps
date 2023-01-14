from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    
    # load test data
    df_test = pd.read_csv(os.path.join(test_data_path, r'testdata.csv'))

    # load model
    with open(os.path.join(model_path, r'trainedmodel.pkl'), 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    pickle_file.close()

    # extract features matrix and target vector from test data
    X_test = df_test.drop(['corporation', 'exited'], axis=1)
    y_test = df_test['exited']

    # model scoring
    y_pred = model.predict(X_test)
    score = metrics.f1_score(y_test, y_pred)
    print("F1 score: ", round(score, 2))

    # save scoring
    with open(os.path.join(model_path, 'latestscore.txt'), 'w') as scoring_file:
        scoring_file.write("f1 score: " + str(round(score, 2)))
    scoring_file.close()

    return score


if __name__ == '__main__':
    score_model()