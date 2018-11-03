#Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing DataSet
trainset = pd.read_csv("train.csv")

variable_set_train = list(trainset.columns.values)

#Dropping unimportant features
trainset_new = trainset.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

#Imputing the missing values
trainset_new.isnull().any()
trainset_new["Age"].fillna(trainset_new["Age"].mean(), inplace= True)
trainset_new['Embarked'].fillna('S', inplace= True)

#Separating IV and DV
X = trainset_new.iloc[:,1:].values
y = trainset_new.iloc[:,0].values

#Handling Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])
X[:,6]= labelencoder_X.fit_transform(X[:,6].astype(str))
onehotencoder_X = OneHotEncoder(categorical_features=[6])
X = onehotencoder_X.fit_transform(X).toarray()

#Splitting the data between Train and validation set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.25)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Using Random Forest Approach to predict results 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#finding optimim value of n_estimators
n = [14,15,16,17,18,20,25,27]
accuracies =[]
for i in n:
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print('Accuracy for {} neighbors is {}'.format(i,round(acc,3)))

#Building model for n_estimators=14
rfc = RandomForestClassifier(n_estimators=14)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
acc_14 = metrics.accuracy_score(y_test, y_pred)
print('Accuracy of the model is {}'.format(round(acc_14,3)))

#Making Confusion Matrix to evaluate model's performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('Confusion Metrics: {}'.format(cm))

#Applying the model on test set
testset = pd.read_csv("test.csv")

variable_set_test = list(testset.columns.values)

testset_new = testset.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

#Imputing the missing values
testset_new.isnull().any()
testset_new["Age"].fillna(testset_new["Age"].mean(), inplace= True)
testset_new["Fare"].fillna(testset_new["Fare"].mean(), inplace= True)
testset_new['Embarked'].fillna('S', inplace= True)

#Creating Feature Matrix
X_test = testset_new.iloc[:,:].values

#Handling Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_test[:,1] = labelencoder_X.fit_transform(X_test[:,1])
X_test[:,6] = labelencoder_X.fit_transform(X_test[:,6].astype(str))
onehotencoder_X = OneHotEncoder(categorical_features=[6])
X_test = onehotencoder_X.fit_transform(X_test).toarray()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test = sc_X.fit_transform(X_test)

#Applying model with n_estimators=14
y_pred = rfc.predict(X_test)

#Returning the labeled data
testset['Survived'] = pd.DataFrame(y_pred)