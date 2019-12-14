import pandas as pd

# If using centroids Dataset
spy = pd.read_csv('./FullSets/C1K+Attacks.csv')

# If using cut down Dataset
# spy = pd.read_csv('./FullSets/D2+Attacks.csv')
# spy = spy.drop(spy.columns[[0, 1, 2, 3]], axis=1)

spy = spy.sample(frac=1) # Shuffle data

p_train = 0.80 
train = spy[:int((len(spy))*p_train)]
test = spy[int((len(spy))*p_train):]

print("Training samples ", len(train))
print("Test Samples: ", len(test))

features = spy.columns.difference(['attack'])
x_train = train[features]
y_train = train['attack']

x_test = test[features]
y_test = test['attack']

X, y = x_train, y_train 

import numpy as np
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(n_estimators=512, n_jobs=-1)

# param_dist = {"max_features": ['auto', 'sqrt'], # Number of features to consider at every split
#               "max_depth": [7,6,5,4,3,2,None], # Maximum number of levels in tree
#               "min_samples_split": sp_randint(2, 50), #  Minimum number of samples required to split a node
#               "min_samples_leaf": sp_randint(1, 50), # Minimum number of samples required at each leaf node
#               "bootstrap": [True, False], # Method of selecting samples for training each tree
#               "criterion": ["gini", "entropy"]}

# random_search = RandomizedSearchCV(clf, scoring= 'f1_micro', 
#                                    param_distributions=param_dist, 
#                                    n_iter= 80)                       

# random_search.fit(X, y)

# def report(results, n_top=3): # Función para mostrar resultados
#     for i in range(1, n_top + 1):
#         candidates = np.flatnonzero(results['rank_test_score'] == i)
#         for candidate in candidates:
#             print("Model with rank: {0}".format(i))
#             print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                   results['mean_test_score'][candidate],
#                   results['std_test_score'][candidate]))
#             print("Parameters: {0}".format(results['params'][candidate]))
#             print("")

# report(random_search.cv_results_)

clf_rf = RandomForestClassifier(n_estimators = 512, criterion = 'entropy', 
                                max_depth=None, max_features = 'auto', 
                                min_samples_leaf = 3, min_samples_split = 4,
                                bootstrap=True, n_jobs=-1, 
                                class_weight=None)

clf_rf.fit(x_train, y_train) # Construcción del modelo

preds_rf = clf_rf.predict(x_test) # Test del modelo

from sklearn.metrics import classification_report

print("Random Forest: \n" 
      +classification_report(y_true=test['attack'], y_pred=preds_rf))

# Confussion Matrix

print("Confussion Matrix:\n")
matriz = pd.crosstab(test['attack'], preds_rf, rownames=['actual'], colnames=['preds'])
print(matriz)

# Variables relevantes

print("Feature Relevance:\n")
print(pd.DataFrame({'Feature': features ,
              'Relevancy': clf_rf.feature_importances_}),"\n")
print("Maximum relevance RF :" , max(clf_rf.feature_importances_), "\n")