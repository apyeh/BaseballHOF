import pandas as pd
import numpy as np
import sklearn as sk
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import hinge_loss
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt
%matplotlib inline

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from scipy.stats import expon

# ==========================================================================================

def print_metrics(y_true, y_preds):
    '''
    Description: print out accuracy, recall, precision, hinge loss, and f1-score of model
    '''
    print "Accuracy: %.4g" % metrics.accuracy_score(y_true, y_preds, normalize=True)
    print "Recall: %.4g" % metrics.recall_score(y_true, y_preds)
    print "Precision: %.4g" % metrics.precision_score(y_true, y_preds)
    print "Hinge loss: %.4g" % metrics.hinge_loss(y_true, y_preds)
    print "F1 score: %.4g" % metrics.f1_score(y_true, y_preds)

# ==========================================================================================


# X includes eras and awards (no AS)
with open('eligible_hitters_X.pkl') as f:
    X = pickle.load(f)

# Load targets
with open('eligible_hitters_y.pkl') as f:
    y = pickle.load(f)


# Split data into training and hold out set
X_train_val, X_holdout, y_train_val, y_holdout \
    = train_test_split(X, y, test_size=0.20, random_state=42)

# Split train_val data into training set and validation set
X_train, X_val, y_train, y_val \
    = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# ==========================================================================================

# Over-sampled data

# Generate the new dataset using under-sampling method
verbose = False
ratio = 'auto'

# 'Random over-sampling'
OS = RandomOverSampler(ratio=ratio, verbose=verbose)
X_train_os, y_train_os = OS.fit_sample(X_train, y_train)

# 'SMOTE'
smote = SMOTE(ratio=ratio, verbose=verbose, kind='regular')
X_train_smo, y_train_smo = smote.fit_sample(X_train, y_train)

# 'SMOTE bordeline 1'
bsmote1 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline1')
X_train_bs1, y_train_bs1 = bsmote1.fit_sample(X_train, y_train)

# 'SMOTE bordeline 2'
bsmote2 = SMOTE(ratio=ratio, verbose=verbose, kind='borderline2')
X_train_bs2, y_train_bs2 = bsmote2.fit_sample(X_train, y_train)

# 'SMOTE SVM'
svm_args={'class_weight': 'auto'}
svmsmote = SMOTE(ratio=ratio, verbose=verbose, kind='svm', **svm_args)
X_train_svs, y_train_svs = svmsmote.fit_sample(X_train, y_train)


# ==========================================================================================

'''
Build models
'''

'''
linear SVM model
'''

# initial linear SVM model
svm01 = LinearSVC()
svm01.fit(X_train, y_train)
svm01_predictions = svm01.predict(X_val)

print_metrics(y_val, svm01_predictions)


# Do grid search of parameters
param_grid_svm02b = {'C': [0.1, 1, 2.5, 5, 7.5, 10, 20, 50]}

svm02b_gscv = GridSearchCV(svm02b, param_grid_svm02b)
svm02b_gscv.fit(X_train_val, y_train_val)

# ==========================================================================================

'''
non-linear SVM model
'''

# initial non-linear SVM model
svmnl01 = SVC()
svmnl01.fit(X_train, y_train)
svmnl01_predictions = svmnl01.predict(X_val)

print_metrics(y_val, svmnl01_predictions)


# ==========================================================================================

'''
Logistic Regression
'''

# initial LR model
logr01_gscv = GridSearchCV(logr01, param_grid_logr01)
logr01_gscv.fit(X_train_val, y_train_val)

# Do grid search of parameters
param_grid_logr01 = {'C': [0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                     'class_weight': [None, 'auto']}

# ==========================================================================================

'''
KNN
'''

# initial KNN model
knn01 = KNeighborsClassifier(n_jobs=-1)
knn01.fit(X_train, y_train)
knn01_preds = knn01.predict(X_val)

print_metrics(y_val, knn01_preds)

# ==========================================================================================

'''
Decision Tree
'''

# initial DT model
dt01 = DecisionTreeClassifier()
dt01.fit(X_train, y_train)
dt01_predictions = dt01.predict(X_val)

print_metrics(y_val, dt01_predictions)

# ==========================================================================================

'''
Decision Tree
'''

# initial GBC model
gbc01 = GradientBoostingClassifier()
gbc01.fit(X_train, y_train)
gbc01_predictions = gbc01.predict(X_val)

print_metrics(y_val, gbc01_predictions)

# ==========================================================================================

'''
Random Forest
'''

# initial model
rf01 = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=275, \
                              class_weight={0: 0.75, 1: 0.25},
                              random_state=23)
rf01.fit(X_train, y_train)
rf01_predictions = rf01.predict(X_val)

print_metrics(y_val, rf01_predictions)

# Train on all data
rf01b = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=275, \
                               class_weight={0: 0.75, 1: 0.25},
                              random_state=23)
rf01b.fit(X_train_val, y_train_val)
rf01b_predictions = rf01b.predict(X_holdout)

print_metrics(y_holdout, rf01b_predictions)


# See feature importances
print sorted(zip(X_train.columns, rf01b.feature_importances_), key=lambda x: x[1], reverse=True)

# Write out model as pickle file
with open('rf01b.pkl', 'w') as f:
    pickle.dump(rf01b, f)

