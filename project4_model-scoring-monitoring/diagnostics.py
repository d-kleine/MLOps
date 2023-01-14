
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(dataset=None):
    #read the deployed model and a test dataset, calculate predictions
    
    if dataset is None:
        dataset_path = os.path.join(test_data_path, 'testdata.csv')
        df = pd.read_csv(dataset_path)
        
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df['exited']    

    modelpath = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
    with open(modelpath, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    
    return y_pred #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    
    finaldata_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(finaldata_path)

    # compute statistics per numeric column
    stats = []
    for col in df.select_dtypes('number').columns:
        stats.append([col + " (mean):", df[col].mean()])
        stats.append([col + " (median):", df[col].median()])
        stats.append([col + " (std):", df[col].std()])
 
    return stats #return value should be a list containing all summary statistics


def missing_data():
    dataset_path = os.path.join(dataset_csv_path, 'finaldata.csv')
    df = pd.read_csv(dataset_path, 'finaldata.csv')

    missing_data = []
    for col in df.columns:
        missing_data.append([col + " (%):", int(df[col].isna().sum() / len(df) * 100)])

    return missing_data

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    
    # timing ingestion step
    timing_measures = []
    for file in ['ingestion.py', 'training.py']:
        start_time = timeit.default_timer()
        os.system('python {}'.format(file))
        end_time = timeit.default_timer()
        duration_step = end_time - start_time
        duration_step_rounded = round(duration_step, 2)
        timing_measures.append(duration_step_rounded)
    
    return timing_measures #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return outdated_packages



if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()

    print(model_predictions())
    print(dataframe_summary())
    print(missing_data)
    print(execution_time())
    print(outdated_packages_list())    
