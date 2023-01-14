import pandas as pd
import os
import json


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
        
    # get root directory
    working_dir = os.getcwd()

    # ingested files placeholder
    ingestedfiles = []
    
    # create final empty dataframe
    columns = ['corporation', 
               'lastmonth_activity',
               'lastyear_activity', 
               'number_of_employees', 
               'exited']
    finaldata = pd.DataFrame(columns=columns)

    # check for datasets
    files_dir = os.listdir(input_folder_path)

    # compile datasets and store ingested file names
    for file in files_dir:
        filepath = os.path.join(input_folder_path,file)
        temp = pd.read_csv(filepath)
        finaldata = pd.concat([finaldata,temp], axis=0)
        ingestedfiles.append(file)

    # drop duplicates
    finaldata.drop_duplicates(inplace=True)
    
    # create output folder for ingested data
    output_dir = os.path.join(working_dir, r'ingesteddata')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
  
     # drop duplicates
    finaldata.drop_duplicates(inplace=True)

    # write dataset to an output file
    savepath = os.path.join(output_folder_path, 'finaldata.csv')
    finaldata.to_csv(savepath, index=False)
    
    # save ingested files
    savepath = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(savepath, 'w') as f:
        f.write(str(ingestedfiles))

if __name__ == '__main__':
    merge_multiple_dataframe()
