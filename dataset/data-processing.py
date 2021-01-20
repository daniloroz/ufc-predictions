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

#find cardinality
for var in df.columns:
    print(var, '\n', df[var].value_counts()/len(df))

#find missing values and percentages
for var in df.columns:
    if df[var].isnull().sum()/len(df) > 0:
        print(var, df[var].isnull().mean().round(3))
#df.drop('tenure_termed', axis=1, inplace=True)

#outlier treatment 
def find_outlier(feature):
    sorted(feature)
    q1,q3 = np.percentile(feature , [25,75])
    IQR = q3 - q1
    lower_range = q1 - (1.5 * IQR)
    upper_range = q3 + (1.5 * IQR)
    return lower_range,upper_range

find_outlier(df['B_losses'])
lower_range, upper_range = find_outlier(df['B_losses'])
df[(df['B_losses'] < lower_range) | (df['B_losses'] > upper_range)]

#training and test splitting
X = df.drop('Win', axis=1)
y = df['Win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

""" ENCODING TECHNIQUES
# impute categorical features with more than 5% missing values w/ a new category 'missing'
process_pipe = make_pipeline(
    mdi.CategoricalVariableImputer(variables=[''], imputation_method='missing'),
    
# Imputing categorical features with less than 5% missing values w/the mode
    mdi.CategoricalVariableImputer(variables=[''], imputation_method='frequent'),
    
# Imputing missing values for numerical feature 'days_since_review' with an arbitrary digit
    mdi.ArbitraryNumberImputer(arbitrary_number = -99999, variables=''),
   
# We are adding a feature to indicate (binary indicator) which records were missing
    mdi.AddMissingIndicator(variables=['']),
    
# Encoding rare categories (less than 1% & the feature must have at least 5 categories)
    ce.RareLabelCategoricalEncoder(tol=0.01, n_categories=5,
                                   variables=['State']),
    
# Encoding rare categories (less than 2% & the feature must have at least 5 categories)
    ce.RareLabelCategoricalEncoder(tol=0.02, n_categories=5,
    variables=['']),
    
# Encoding rare categories (less than 5% & the feature must have at least 5 categories)
    ce.RareLabelCategoricalEncoder(tol=0.05, n_categories=5,
                                   variables=['']),
    
# Target or Mean encoding for categorical features
    ce.OrdinalCategoricalEncoder(encoding_method='ordered',
    variables=[''] 
))"""

#final processing
process_pipe.fit(X_train, y_train)
X_train_clean = process_pipe.transform(X_train)
X_test_clean = process_pipe.transform(X_test)
X_train_clean.head()
