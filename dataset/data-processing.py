import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from feature_engine import imputation as mdi
from feature_engine import encoding as ce
from sklearn.model_selection import train_test_split


#load our data into a dataframe object
with open('data.csv') as f:
    df = pd.read_csv(f)
f.close()
df.head()
df.info()

#determine feature variability. See if we can delete any values with low variance
for col in df.columns:
    print(col, df[col].nunique(), len(df))

#remove unique features afetr analysis on variance
df.drop(['R_fighter'], axis=1, inplace=True)
df.drop(['B_fighter'], axis=1, inplace=True)
df.drop(['Referee'], axis=1, inplace=True)
df.drop(['location'], axis=1, inplace=True)

