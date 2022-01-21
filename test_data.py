#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import scipy.stats


# In[2]:


def test_row_count():
    data=pd.read_csv("./amazondrive/census.csv")
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


# In[ ]:




