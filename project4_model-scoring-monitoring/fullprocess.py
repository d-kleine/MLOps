import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import os
import sys
import ast
import subprocess
import json
import pandas as pd
import pickle
import logging
from sklearn.metrics import f1_score

# initialize logging
logging.basicConfig(filename='journal.log',
                    level=logging.INFO,
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')

with open('config.json', 'r') as json_file:
    config = json.load(json_file)
json_file.close()

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_model_path = os.path.join(config['output_model_path']) 
output_folder_path = os.path.join(config['output_folder_path']) 

next_step = False

def full_process():
    
##################Check and read new data
#first, read ingestedfiles.txt
    with open(os.path.join(os.path.join(prod_deployment_path, "ingestedfiles.txt"))) as ingested_file:
        ingested_files = ingested_file.read().splitlines()[-1]
    ingested_file.close()

#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    dataset_files = os.listdir(os.path.join(input_folder_path))
    if (ingested_files == dataset_files):
            logging.info('No new data found - exiting')
            sys.exit()
    else:
        logging.info('New data found - continuing')
        ingestion.merge_multiple_dataframe()
        next_step = True

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    if next_step:
        scorespath = os.path.join(prod_deployment_path, 'latestscore.txt')
        with open(scorespath, 'r') as f:
             latest_score = f.read()
                
        latest_score_int = float(latest_score.split(': ')[-1])

        filepath = os.path.join(output_folder_path,'finaldata.csv')
        df_new = pd.read_csv(filepath)
        
        X_new = df_new.drop(['corporation', 'exited'], axis=1)
        y_new = df_new['exited']    
        
        model_path_old = os.path.join(prod_deployment_path, 'trainedmodel.pkl')
        with open(model_path_old, 'rb') as f:
            model = pickle.load(f)
        y_pred_new = model.predict(X_new)
        new_score = f1_score(y_new, y_pred_new)
        print(f'latest score: {latest_score_int}, new score: {new_score}')
     
        if (latest_score_int >= new_score):
            logging.info('No model drift - ending process')
            next_step = False
            sys.exit()         
        else:
            logging.info('Model drift - continuing')
            next_step = True

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    if next_step:
        logging.info('Training new model')
        training.train_model()
        
        logging.info('Scoring new model on test data')
        scoring.score_model()
        
        logging.info('Deploying new model')
        deployment.store_model_into_pickle(model) # = os.system("python deployment.py")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
        os.system("python diagnostics.py")
        os.system("python reporting.py")
        os.system("python apicalls.py")

if __name__ == '__main__':
    full_process()
    