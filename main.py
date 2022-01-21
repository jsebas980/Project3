#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
from ml.data import process_data
from model.model import inference
from pydantic import BaseModel, Field
import os
import pandas as pd
import pickle


# In[2]:


# Instantiate the app.
app = FastAPI()


# In[3]:


#Load Models
result_model = pickle.load(open("ml/finalized_model.sav", 'rb'))
result_encoder = pickle.load(open("ml/OneHotEncoder.sav", 'rb'))
result_lb = pickle.load(open("ml/LabelBinarizer.sav", 'rb'))


# In[4]:


class TaggedItem(BaseModel):
    age: int
    workclass: str
    fnlgt:int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week") 
    native_country: str = Field(alias="native-country")
    class Config:
        schema_extra = {
        "example": {
            "age": 20,
            "workclass": "Private",
            "fnlgt": 168187,
            "education": " Some-college",
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
    }








# In[5]:


def convert_input_data(record):
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
    #df_record=pd.DataFrame(record)
    array_converted={"position":record}
    print(array_converted)
    df_record= pd.DataFrame.from_dict(array_converted,orient='index')
    X, y,encoder,lb = process_data(
            df_record, categorical_features=cat_features,training=False, encoder=result_encoder, lb=result_lb,
            )
    print("Prueba")
    return X


# In[6]:


# Define a GET on the specified endpoint.
@app.get("/")
async def initial_message():
    welcome_message="Greetings, Welcome to the api for module 3 of course Machine learning devops engineer"
    return welcome_message


# In[9]:


@app.post("/predict/")
async def predict_salary(item: TaggedItem):
    print(item.dict())
    X=convert_input_data(item.dict(by_alias=True))
    prediction=inference(result_model,X)
    category=result_lb.inverse_transform(prediction[0])
    final_result= "The prediction of Salary according to the features is:" + category[0]
    return final_result


# In[ ]:


import requests
import json

def test_get_method():
    path = "http://127.0.0.1:8000/"
    response = requests.get(url=path)
    responseJson = json.loads(response.text)
    assert response.status_code == 200



if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    #os.system("rm -r .dvc .apt/usr/lib/dvc")




