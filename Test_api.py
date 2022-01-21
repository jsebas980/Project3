#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import json
import pytest


# In[4]:


def test_get():
    response = requests.get("http://127.0.0.1:8000")
    assert response.status_code == 200


# In[5]:


def test_post_predict():
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
    response=requests.post(url="http://127.0.0.1:8000/predict", json=body)
    assert response.status_code == 200


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
    response=requests.post(url="http://127.0.0.1:8000/predict", json=body)
    assert response.status_code == 200


# In[ ]:




