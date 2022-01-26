#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import json
import pytest


# In[4]:


def test_get():
    response = requests.get("https://projectjs3.herokuapp.com/")
    assert response.status_code == 200
    assert response.json() == [0]

# In[5]:


def test_post_predict():
    body={
            "age": 20,
            "workclass": "Private",
            "fnlgt": 168187,
            "education": "Some-college",
            "education-num": 10,
            "marital-status": "Never-married",
            "occupation": "Other-service",
            "relationship": "Other-relative",
            "race": "White",
            "sex": "Female",
            "capital-gain": 4416,
            "capital-loss": 0,
            "hours-per-week": 25,
            "native-country": "United-States"
        }
    response=requests.post(url="https://projectjs3.herokuapp.com/predict", json=body)
    assert response.status_code == 200
    assert response.json() == "The prediction of Salary according to the features is: >50K"

# In[6]:


def test_post_predict2():
    body={
            "age": 34,
            "workclass": "Federal-gov",
            "fnlgt": 337895,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Married-civ-spouse",
            "occupation": "Prof-specialty",
            "relationship": "Husband",
            "race": "Black",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        }
    response=requests.post(url="https://projectjs3.herokuapp.com/predict", json=body)
    assert response.status_code == 200
    assert response.json() == "The prediction of Salary according to the features is: <=50K"






