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

#confusion matrix
cm = metrics.confusion_matrix(y_test, predictions)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
