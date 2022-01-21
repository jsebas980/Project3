import requests
import json
import pytest


def predict():
    body={
            "age": 34,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 292175,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Divorced",
            "occupation": "Exec-managerial",
            "relationship": "Unmarried",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 45,
            "native-country": "United-States"
        }
    response=requests.post(url="https://projectjs3.herokuapp.com/predict", json=body)
    print(f"Status Code:   {response.status_code}")
    print (f"The Prediction is:  + {response.text}")

if __name__ == "__main__":
    predict()