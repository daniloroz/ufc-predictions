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
df.drop(['date'], axis=1, inplace=True)
df.drop(['title_bout'], axis=1, inplace=True)


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

""" ENCODING TECHNIQUES """
# impute categorical features with more than 5% missing values w/ a new category 'missing'
process_pipe = make_pipeline(
    mdi.CategoricalImputer(variables=['R_Stance', 'B_Stance'], imputation_method='missing'),
                             

  mdi.ArbitraryNumberImputer(arbitrary_number = 30, variables=['R_age', 'B_age']),

# Imputing missing values for numerical feature 'days_since_review' with an arbitrary digit
    mdi.ArbitraryNumberImputer(arbitrary_number = 0.0, variables=['B_avg_BODY_att', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_DISTANCE_att', 'B_avg_DISTANCE_landed', 'B_Reach_cms', 'R_Reach_cms', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_LEG_landed', 'B_avg_PASS', 'B_avg_REV', 'B_avg_SIG_STR_att', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed', 'B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_att', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TD_pct', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed',
                                                                  'B_total_time_fought(seconds)', 'R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_KD', 'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_PASS', 'R_avg_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_avg_TOTAL_STR_att', 'R_avg_TOTAL_STR_landed', 
                                                                  'R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_opp_DISTANCE_att', 'R_avg_opp_DISTANCE_landed', 'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed', 'R_avg_opp_KD', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 'R_avg_opp_PASS', 'R_avg_opp_REV', 'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_opp_SIG_STR_pct', 'R_avg_opp_SUB_ATT', 'R_avg_opp_TD_att', 'R_avg_opp_TD_landed', 'R_avg_opp_TD_pct', 'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed', 'R_total_time_fought(seconds)'
                                                                  ]),
    
# Target or Mean encoding for categorical features
    ce.OrdinalEncoder(encoding_method='ordered', variables=['weight_class', 'R_Stance', 'B_Stance'] 
))

#final processing
process_pipe.fit(X_train, y_train)
X_train_clean = process_pipe.transform(X_train)
X_test_clean = process_pipe.transform(X_test)
X_train_clean.head()
