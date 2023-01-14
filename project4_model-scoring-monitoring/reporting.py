import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    
    # collect test dataset and target vector
    dataset_path = os.path.join(test_data_path, 'testdata.csv')
    df_test = pd.read_csv(dataset_path)
    y = df_test['exited']
    
    # perform prediction
    y_pred = model_predictions()
    
    # calculate confusion matrix
    cm = metrics.confusion_matrix(y, y_pred)
    
    # Create cm plot
    fig, ax = plt.subplots()
    sns.heatmap(data=cm, annot=True)
    ax.set_xlabel('Predicted')
    ax.xaxis.set_ticklabels(['not churned', 'churned'])
    ax.set_ylabel('True')
    ax.yaxis.set_ticklabels(['not churned', 'churned'])
    plt.title('Confusion matrix')
    
    # write the confusion matrix to the workspace
    savepath = os.path.join(model_path,'confusionmatrix.png')
    fig.savefig(savepath)

if __name__ == '__main__':
    score_model()
