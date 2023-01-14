from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    
    # get root directory
    working_dir = os.getcwd()
    
    # import final data
    filepath = os.path.join(dataset_csv_path, "finaldata.csv")
    df_train = pd.read_csv(filepath)
    
    # extract feature matrix and target vector
    X_train = df_train.drop(["corporation", "exited"], axis=1)
    y_train = df_train["exited"]    
    
    # fit the logistic regression to your data
    model.fit(X_train, y_train)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    
    # create output folder for dumped model
    output_dir = os.path.join(working_dir, r'practicemodels')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    
    # save model dump
    with open(os.path.join(model_path,'trainedmodel.pkl'), 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    pickle_file.close()

    
if __name__ == '__main__':
    train_model()