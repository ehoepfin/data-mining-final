"""
File name: 4380_FinalProject.py
@author: Jake Hobbs
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# reading in the data
crop_loss = pd.read_csv("FinalProjectDataset.csv", header = 0,
                        skipinitialspace = True)

##############################
##### DATA PREPROCESSING #####
##############################

# stripping whitespaces and removing quantity types not measured in acres
crop_loss['Quantity Type'] = crop_loss['Quantity Type'].str.strip()
crop_loss['Coverage Category'] = crop_loss['Coverage Category'].str.strip()

crop_loss = crop_loss.loc[crop_loss['Quantity Type'] == 'Acres']

# dropping columns that have corresponding numeric codes
crop_loss.drop(['Commodity Year', 'Location State Abbreviation',
                'Location County Name', 'Commodity Name',
                'Insurance Plan Name', 'Quantity Type'], 1, inplace = True)

# one-hot encoding categorical variables
oh_insurance = pd.get_dummies(crop_loss['Insurance Plan Code'],
                             prefix = 'Insurance')
crop_loss.drop('Insurance Plan Code', axis = 1, inplace = True)
crop_loss = crop_loss.join(oh_insurance)

oh_coverage = pd.get_dummies(crop_loss['Coverage Category'],
                             prefix = 'Coverage')
crop_loss.drop('Coverage Category', axis = 1, inplace = True)
crop_loss = crop_loss.join(oh_coverage)

oh_delivery = pd.get_dummies(crop_loss['Delivery Type'], prefix = 'Delivery')
crop_loss.drop('Delivery Type', axis = 1, inplace = True)
crop_loss = crop_loss.join(oh_delivery)

# scaling the data using the standard scaler
scaler = StandardScaler()

X = crop_loss.drop('Indemnity Amount', 1)
X_S = scaler.fit_transform(X)

y = crop_loss['Indemnity Amount']

# splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_S, y, test_size = 0.25,
                                                    shuffle = True)

##########################
##### MODEL CREATION #####
##########################

### Multiple Linear Regression ###
from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(X_train, y_train)       # fit lm to the training sets

mlr_pred = mlr.predict(X_test)  # make predictions


### Bagging ###
from sklearn import tree
from sklearn.ensemble import BaggingRegressor

base_estimator = tree.DecisionTreeRegressor()
bagging = BaggingRegressor(base_estimator = base_estimator,
                           n_estimators=25,
                           max_samples=1.0,
                           max_features=1.0)

bagging.fit(X_train, y_train)       # fit br to the training sets

bag_pred = bagging.predict(X_test)  # make predictions


### Neural Network ###
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# creating the optimizer for the network
opt = Adam(learning_rate = 0.005, beta_1 = 0.9, beta_2 = 0.999,
           epsilon = 1e-07, amsgrad = False, name = 'Adam')

# initializing and adding the layers to the model
NN = Sequential()
NN.add(Dense(42, input_dim = 42, kernel_initializer = 'normal',
             activation = 'relu'))
NN.add(Dense(1, kernel_initializer = 'normal'))
NN.compile(loss = 'mean_squared_error', optimizer = opt, metrics = 'accuracy')

# fitting the model to the data
NN.fit(X_train, y_train, epochs = 250, batch_size = 500, verbose=1)

# making predictions
NN_pred = NN.predict(X_test)

############################
##### MODEL EVALUATION #####
############################
from sklearn.metrics import mean_squared_error, r2_score

print('-'*10, 'MODEL EVALUATION', '-'*10)
"""
Model Evaluation - Multiple Linear Regression
Metrics: MSE, R^2
"""
print('\n***Multiple Linear Regression***')
mlr_mse = mean_squared_error(y_test, mlr_pred)
print('Mean Squared Erorr: ', mlr_mse)
mlr_rsq = r2_score(y_test, mlr_pred)
print('R^2: ', mlr_rsq)

"""
Model Evalutation - Bagging Regressor
Metrics - MSE, R^2
"""
print('\n***Bagging Regressor***')
bag_mse = mean_squared_error(y_test, bag_pred)
print('Mean Squared Erorr: ', bag_mse)
bag_rsq = r2_score(y_test, bag_pred)
print('R^2: ', bag_rsq)

"""
Model Evaluation - Neural Network
Metrics: MSE, R^2
"""
print('\n***Neural Network***')
print(NN.summary())
NN_mse = mean_squared_error(y_test, NN_pred)
print('Mean Squared Erorr: ', NN_mse)
NN_rsq = r2_score(y_test, NN_pred)
print('R^2: ', NN_rsq)
