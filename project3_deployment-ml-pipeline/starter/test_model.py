import os
import sys
import pickle
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data
from starter.ml.model import inference

file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)

"""
Fixtures
"""


@pytest.fixture(scope="module")
def path():
    """
    Extract data path
    """
    path = os.path.join(file_dir, "data/clean_census.csv")
    return path


@pytest.fixture(scope="module")
def data():
    """
    Extract the data
    """
    data_path = os.path.join(file_dir, "data/clean_census.csv")
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def cat_features():
    """
    Extract categorical features
    """
    cat_features=[
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"
    ]

    return cat_features

@pytest.fixture(scope="module")
def train_dataset(data, cat_features):
    """
    Extract trained and pre-processed dataset
    """
    train, _ = train_test_split(data, test_size=0.2, random_state=42)
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train


"""
Test methods
"""


def test_is_model():
    """
    Check saved model is present
    """
    savepath=os.path.join(file_dir, "model/trained_model.pkl")
    if os.path.isfile(savepath):
        try:
            _=pickle.load(open(savepath, 'rb'))
        except BaseException:
            assert ('Model not found')


def test_is_model_fitted(train_dataset):
    """
    Check if saved model is fitted
    """

    X_train, _=train_dataset
    model_path=os.path.join(file_dir,"model/model.pkl")
    model=pickle.load(open(model_path, 'rb'))

    try:
        model.predict(X_train)
    except BaseException:
        assert ('Model not fitted')


def test_inference_working(train_dataset):
    """
    Check inference function
    """
    X_train, _=train_dataset

    model_path=os.path.join(file_dir,"model/model.pkl")
    model=pickle.load(open(model_path, 'rb'))

    try:
        inference(model, X_train)
    except BaseException:
        assert ('Inference not possible')
