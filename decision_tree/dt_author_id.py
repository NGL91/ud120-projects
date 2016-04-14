#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree

cf = tree.DecisionTreeClassifier(min_samples_split=40)


to=time()
cf = cf.fit(features_train, labels_train)


print "\n Time to train in Decision Tree:",time()-t0

t1=time()
pred = cf.predict(features_test)

print "\n Time to predict in Decision Tree:",time()-t1


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print accuracy
#0.978384527873
#########################################################


