# Model Details
## Juan Sebastian Jaramillo created the model. It is logistic regression using the followings hypeparameters: max_iter=100, solver='lbfgs', fit_intercept=True, intercept_scaling=1 with the objective to obtain better results, the versión of the libreary is scikit-learn 1.0.2.

## Intended Use
### This model should be used to predict the income if greather 50K per year base don Census data.  

## Metrics
### The model was evaluated using Accuracy 0.92 Recall 0.21 and FBeta with 0.34

## Data
### The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) . 
### Extraction was done by Barry Becker from the 1994 Census database. 

### Prediction task is to determine whether a person makes a salary over 50K a year.
### The original data set has 32561 rows, 15 columns, cleaned dataset contains 30162 rows, 15 columns. 10 cross validation split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features.

## Bias
### This model represents in majority part of categorical sex Mens, who lives in the United States, and race White, the data is unbalanced, specially for objective variable salary for this reason it’s necesary to balance the data because in original form the model is not 100% reliable

## Ethical Considerations
### There are categories where the data is clearly biased, so it is necessary to have a better balance of the information at the category level in order to avoid biases that the model may have for decision making.  

## Caveats & Recommendations
### The purpose of this dataset is for academic use. It is not recommended to use it as an official database or to use it as a reference for any productive model, since in some classes and categories there is clearly an overfitting, so this overfitting could cause problems for data collection. decisions 