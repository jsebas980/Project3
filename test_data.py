#!/usr/bin/env python
# coding: utf-8

# In[3]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ml.data import process_data
from model.model import train_model, compute_model_metrics
import pandas as pd
import numpy as np
import scipy.stats
import pickle

# In[2]:


def test_row_count():
    data=pd.read_csv("./amazondrive/Census_cleaned.csv")
    assert 30000 < data.shape[0] < 150000


# In[ ]:


def test_column_names():
    data=pd.read_csv("./amazondrive/Census_cleaned.csv")
    expected_colums = [
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    ]
    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


# In[4]:


def test_relationship():
    data=pd.read_csv("./amazondrive/Census_cleaned.csv")
    known_categories = [" Not-in-family", " Husband", " Wife", " Own-child", " Unmarried", " Other-relative"]

    relation = set(data['relationship'].unique())

    # Unordered check
    assert set(known_categories) == set(relation)


def test_precision():
    file = 'ml/finalized_model.sav'
    model = pickle.load(open(file, 'rb'))

    data=pd.read_csv('./amazondrive/Census_cleaned.csv')

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    X, y, encoder, lb = process_data(
    data, categorical_features=cat_features, label="salary", training=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    preds=model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test,preds)

    assert precision > 0.1






