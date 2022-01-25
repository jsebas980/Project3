#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from ml.data import process_data
from model.model import train_model, compute_model_metrics
import pandas as pd
import pickle



# Add code to load in the data.
def train_model():
    census_data=pd.read_csv("./amazondrive/Census_cleaned.csv")

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    #train, test = train_test_split(data, test_size=0.20)

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

    kf10 = KFold(n_splits=10, shuffle=False)
    label="salary"
    X, y, encoder, lb= process_data(census_data,cat_features,label,training=True)

    for train_index, test_index in kf10.split(census_data):
        X_train,X_test = X[train_index], X[test_index]
        y_train,y_test = y[train_index], y[test_index]
        model=train_model(X_train,y_train)
        prediction=model.predict(X_test)
        precision, recall, fbeta=compute_model_metrics(y_test, prediction)
        print("Precision: ",precision,", Recall: ", recall, ", fbeta: ", fbeta)


    #Train model using all the data 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    model=train_model(X_train, y_train)

    prediction=model.predict(X_test)
    precision, recall, fbeta=compute_model_metrics(y_test, prediction)
    pickle.dump(lb, open('ml/LabelBinarizer.sav', 'wb'))
    pickle.dump(encoder, open('ml/OneHotEncoder.sav', 'wb'))
    pickle.dump(model, open('ml/finalized_model.sav', 'wb'))



    #Function for calculating descriptive stats on slices of the dataset

    def slice_data(df, cat_features, encoder, lb, model):
        for cat in cat_features:
            print("Category: ", cat)
            for cls in df[cat].unique():
                print(cls)
                df_temp = df[df[cat] == cls]
            
                X_temp, y_temp, encoder1, lb1 = process_data(
                df_temp, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb,
                )
            
                preds=model.predict(X_temp)
    
                precision, recall, fbeta=compute_model_metrics(y_temp, preds)
        
                with open("ml/slice_output.txt", 'a') as f:
                    f.write("\nCategory: "+ cat +", "+ cls+"\n")
                    f.write(" -Precision: " + str(precision)+"\n")
                    f.write(" -Recall: " + str(recall)+"\n")
                    f.write(" -Fbeta: " + str(fbeta)+"\n")


    slice_data(census_data,cat_features,encoder,lb,model)


if '__main__' == __name__:
    train_model()



