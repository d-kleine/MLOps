# library doc string
'''
Python module, which:
    - imports df
    - perform EDA
    - perform feature engineering
    - generate and save plots
    - train models
'''

# import libraries
import os

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(f'{pth}')
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            data_frame: pandas dataframe
    output:
            None
    '''
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    # Churn Histogram
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # Customer_Age Histogram
    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # Marital_Status distribution bar
    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.close()

    # Total_Trans_Ct histrogram
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_trans_Ct.png')
    plt.close()

    # Feature Correlations
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        data_frame.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional] used for naming variables or y column
    output:
            data_frame: pandas dataframe with new columns for
    '''
    # Creating new features based on churn for cat_columns:
    for cat in category_lst:
        feature_lst = []
        feature_groups = data_frame.groupby('Gender').mean()[response]

        for val in data_frame['Gender']:
            feature_lst.append(feature_groups.loc[val])

        data_frame[f'{cat}_{response}'] = feature_lst

    return data_frame


def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional] used for naming variables or y column
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # check for all cat columns:
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    # Feature engineering (mean based data encoding):
    data_frame = encoder_helper(data_frame, category_lst, response)

    # Create empty data_frame
    x_data = pd.DataFrame()
    y_data = data_frame[response]

    # columns to keep for training:
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    x_data[keep_cols] = data_frame[keep_cols]

    # train test split:
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    '''

    # Random Forest results
    plt.figure(figsize=(8, 8))
    # plt.rc('figure', figsize=(15, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    # Logistic Regression results
    plt.figure(figsize=(8, 8))
    # plt.rc('figure', figsize=(15, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    # calculate feature importance:
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # save feature importance plot:
    plt.savefig(f'{output_pth}/feature_importances.png')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # RandomForest:
    rfc = RandomForestClassifier(random_state=42)

    # grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    # LogisticRegression:
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Save roc_curve plot:
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_,
                   x_test, y_test, ax=axis, alpha=0.8)
    plt.figure(figsize=(15, 8))
    plot_roc_curve(cv_rfc, x_test, y_test, ax=axis, alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close()

    # Save classification_score plots
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save feature_importance plot
    feature_importance_plot(cv_rfc, x_train, './images/results/')


if __name__ == '__main__':
    # read data
    DATA = import_data("./data/bank_data.csv")

    # perform EDA and store plots in images
    perform_eda(DATA)

    # process feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DATA, 'Churn')

    # train models
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
