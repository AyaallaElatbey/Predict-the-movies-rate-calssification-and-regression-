import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import preProcessing as preProc
import pickle

# dataset = preProc.preProcessing()
dataset = np.array(pd.read_csv('Preproceced_data.csv'))
X = dataset[:,:-1]
Y = dataset[:,-1]

X = np.c_[np.ones((X.shape[0], 1)), X]
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2, random_state=0,shuffle=True)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
# save the model to disk
filename = 'multiRegression.sav'
pickle.dump(model, open(filename, 'wb'))
# model = pickle.load(open('multiRegression.sav', 'rb'))
y_predict = model.predict(X_test)
print("Multilinear Regression")
print("Data:", dataset.shape)
print("Train set size: ", len(X_train))
print("Test set size: ", len(X_test))
print("Mean absolute error: %.5f" % np.mean(np.absolute(y_predict - y_test)))
print("Residual sum of squares (MSE): %.5f" % np.mean((y_predict - y_test) ** 2))
print("R2-score: %.5f" % r2_score(y_test , y_predict) )
