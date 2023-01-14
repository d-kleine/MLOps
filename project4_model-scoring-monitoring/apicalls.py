import os
import json
import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f)
    
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path']) 
filepath = os.path.join(test_data_path,'testdata.csv')

#Call each API endpoint and store the responses
response1 = requests.post(URL + '/prediction' + f'?filename={filepath}').text #put an API call here
response2 = requests.get(URL + '/scoring').text #put an API call here
response3 = requests.get(URL + '/summarystats').text #put an API call here
response4 = requests.get(URL + '/diagnostics').text #put an API call here

#combine all API responses
responses = {'Predictions': response1,
             'Scoring': response2,
             'Statistics': response3,
             'Diagnostics': response4} #combine reponses here

#write the responses to your workspace
textpath = os.path.join(model_path,'apireturns.txt')
with open(textpath,'w') as f:
    f.write(json.dumps(responses))
