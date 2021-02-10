import numpy as np
import pandas as pd
import csv
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

with open('dataset/preprocessed_data.csv') as f:
    df = pd.read_csv(f)
f.close()
df.head()
df.info()

#training and test splitting
y = df['Winner']
X = df.drop('Winner', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# all parameters not specified are set to their defaults
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

score = model.score(X_test, y_test)
print(score)

#evaluation of current model cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold

lr_scores = cross_val_score(model, X_test_clean, y_test, cv=5)
print(lr_scores)

#validat method
#It allows specifying multiple metrics for evaluation.
#It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.
scoring = ['precision_macro', 'recall_macro']
lr2_scores = cross_validate(model, X_test_clean, y_test, scoring=scoring)
sorted(lr2_scores.keys())
lr2_scores['test_recall_macro']

#kfold
kf = KFold(n_splits=5)
for train, test in kf.split(X):
  print("%s %s" % (train, test))
