# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies credit card customers that are most likely to churn. It includes a Python package for a ML project that follows coding (PEP 8) and engineering best practices for implementing software (modular, documented, and tested).

File struture: 
    data (datasets)
        bank_data.csv (churn dataset)
    images (visualizations)
        eda (EDA results)
            churn_distribution.png (distribution of churn)
            customer_age_distribution.png (distribution of customer_age)
            heatmap.png (correlation matrix)
            marital_status_distribution.png (distribution of marital_status)
            total_transaction_distribution.png (distribution of total_transaction)
        results (model results)
            feature_importance.png (feature importances)
            logistics_results.png (model summary logistic regression)
            rf_results.png (model summary random forest)
            roc_curve_result.png (ROC curve)
    logs (log of code running)
        churn_library.log
    models (model pickles)
        logistic_model.pkl (best model for logistic regression)
        rfc_model.pkl (best model for random forest)
    churn_library.py (library of refactored function for loading the data, cleaning, feature engineering, model training, scoring and results reporting)
    churn_notebook.ipynb (original file for loading the data, cleaning, feature engineering, model training, scoring and results reporting)
    churn_script_logging_and_tests.py (tests for functions implemented in churn_library.py)
    README.md


    data - Contains the data
    enviroment - 
    images/eda - 
    images/results - 
    logs/run_log.log - 
    logs/test_churn_library.log - Log of tests
    models - model pickles
    churn_library.py - contains the 
    churn_script_logging_and_tests.py - contains the 
    requirement.txt - required packages


## Files and data description
The corresponding dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) and stored as *./data* and the path updated in project config file:

    `data_file_path = './data/BankChurners.csv'`

## Running Files
* Install dependencies

Python 3.6:
```bash
pip install -r requirements_py3.6.txt
```

Python 3.8
```bash
pip install -r requirements_py3.8.txt
```

for Python 

* Run script

```bash
python churn_library.py
```
