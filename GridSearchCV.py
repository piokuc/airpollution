import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Train set has these columns:
# Id,region,date,mortality_rate,O3,PM10,PM25,NO2,T2M

# Test set: as above, except it does not have mortality_rate
features = ['O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now

data_dir = '.'

train = pd.read_csv(data_dir + os.sep + 'train.csv', parse_dates = ['date'])
test = pd.read_csv(data_dir + os.sep + 'test.csv')

# We have missing values for NO2 and PM25 for 2007.
# Simplest way of dealing with it is to 
# remove the rows with missing values:
train = train.dropna(axis=0, how = 'any') 

X_train = train[features]
y = train['mortality_rate'].copy()

X_test = test[features].copy()

"""
lr = LinearRegression()
lr.fit(X_train, y)
"""
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 5, 8, 10]}
svr = svm.SVR()
""" http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
"""
model = GridSearchCV(svr, parameters, n_jobs = 4, verbose = 99)
print "best model:", model
print dir(model)

model.fit(X_train, y)

predictions = test[['Id']].copy()
predictions['mortality_rate'] = model.predict(X_test)

predictions.to_csv('svr.csv', index = False)
