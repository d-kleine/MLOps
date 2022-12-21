# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project identifies credit card customers that are most likely to churn. It includes a Python package for a ML project that follows coding (PEP 8) and engineering best practices for implementing software (modular, documented, and tested).

File struture:
* [data/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/data) (datasets)
  * [bank_data.csv](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/data/bank_data.csv) (churn dataset)
* [images/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images) (visualizations)
  * [eda/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda) (EDA results)
    * [churn_distribution.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda/churn_distribution.png) (distribution of churn)
    * [customer_age_distribution.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda/customer_age_distribution.png) (distribution of customer_age)
    * [heatmap.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda/heatmap.png) (correlation matrix)
    * [marital_status_distribution.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda/marital_status_distribution.png) (distribution of marital_status)
    * [total_trans_Ct.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/eda/total_trans_Ct.png) (distribution of total_transaction)
  * [results/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/results) (model results)
    * [feature_importances.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/results/feature_importances.png) (feature importances)
    * [logistic_results.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/results/logistic_results.png)  (model summary logistic regression)
    * [rf_results.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/results/rf_results.png) (model summary random forest)
    * [roc_curve_result.png](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/images/results/roc_curve_result.png) (ROC curve)
* [logs/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/logs) (log of code running)
  * [churn_library.log](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/logs/churn_library.log) (log of *churn_library.py*)
* [models/](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/models) (pickles for best models)
  * [logistic_model.pkl](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/models/logistic_model.pkl) (best model for logistic regression)
  * [rfc_model.pkl](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/models/rfc_model.pkl) (best model for random forest)
* [churn_library.py](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/churn_library.py)  (library of refactored function for loading the data, cleaning, feature engineering, model training, scoring and results reporting)
* [churn_notebook.ipynb](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/churn_notebook.ipynb) (original file for loading the data, cleaning, feature engineering, model training, scoring and results reporting)
* [churn_script_logging_and_tests.py](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/churn_script_logging_and_tests.py) (tests for functions implemented in churn_library.py)
* [Guide.ipynb](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/Guide.ipynb) (guide to run the project)
* [README.md](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/README.md) (documentation of project)
* [requirements_py3.6.txt](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/requirements_py3.6.txt) (files necessary for Python 3.6 to run projekt successfully)
* [requirements_py3.8.txt](https://github.com/d-kleine/Udacity_MLOps/blob/main/project1_clean_code/requirements_py3.8.txt) (files necessary for Python 3.8 to run projekt successfully)


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
