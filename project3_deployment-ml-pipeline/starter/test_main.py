from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World!"


def test_predict_zero():  # post sample: <=50k:
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    response = client.post('/inference', json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": '<=50K'}


def test_predict_one():  # post sample: >50k:
    data = {
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "India"
    }

    response = client.post('/inference', json=data)
    assert response.status_code == 200
    assert response.json() == {"prediction": '>50K'}
