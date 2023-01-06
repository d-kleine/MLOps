import os
import pickle
from pathlib import Path
import pandas as pd

from ml.data import process_data
from ml.model import compute_model_metrics, inference


def slice_metrics(
        model,
        encoder,
        lb,
        data,
        slice_feature,
        categorical_features=[]):
    """
    Output the performance of the model on slices of the data
    Inputs
    ------
    model : Machine learning model
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    data : pd.DataFrame
        Dataframe containing the features and label.
    slice_feature: str
        Name of the feature used to make slices (categorical features)
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])

    Returns
    -------
    None
    """

    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=False, encoder=encoder, lb=lb)
    preds = inference(model, X)

    with open('slice_output.txt', 'w') as f:
        for slice_value in data[slice_feature].unique():
            slice_index = data.index[data[slice_feature] == slice_value]
            
            f.write(str(slice_feature)+' = '+str(slice_value)+'\n')
            f.write('data size:{}\n'.format(len(slice_index)))
            f.write('precision: {}, recall: {}, fbeta: {}\n'.format(
                *compute_model_metrics(y[slice_index], preds[slice_index])
            )
            )
            f.write('--------------------------------------------------------------------------------------------'+'\n')
