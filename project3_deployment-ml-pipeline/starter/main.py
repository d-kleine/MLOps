# Put the code for your API here.
import os
import pandas as pd
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

file_dir = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(file_dir, 'model/model.pkl')
encoder_path = os.path.join(file_dir, 'model/encoder.pkl')
lb_path = os.path.join(file_dir, 'model/label_binarizer.pkl')

model = pickle.load(open(model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
lb = pickle.load(open(lb_path, 'rb'))

# Declare the data object with its components and their type.
# see https://archive.ics.uci.edu/ml/datasets/census+income


class Attributes(BaseModel):
    age: int = Field(None, example=39)
    workclass: str = Field(None, example='State-gov')
    fnlgt: int = Field(None, example=77516)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=13)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Adm-clerical')
    relationship: str = Field(None, example='Not-in-family')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=2174)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=40)
    native_country: str = Field(None, example='United-States')


# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.


@app.get("/")
async def say_hello():
    return "Hello World!"

# Use POST action to send data to the server


@app.post("/inference")
async def predict(sample: Attributes):
    # create DataFrame
    df_data = pd.DataFrame.from_dict([sample.dict(by_alias=True)])

    # categorial features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    # Prepare data for inference
    X, _, _, _ = process_data(
        df_data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run model
    pred = inference(model, X)[0]

    # Get prediction
    if pred == 0:
        pred = "<=50K"
    else:
        pred = ">50K"

    return {"prediction": pred}
