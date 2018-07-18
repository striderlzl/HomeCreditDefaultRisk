import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

#Training data
app_train = pd.read_csv('data/all/application_train.csv')

#Test data
app_test = pd.read_csv('data/all/application_test.csv')

#Absolute value correlation matrix
app_train_corrs = app_train.corr().abs()

#Threshold for removing correlated variables
threshold = 0.9

#Upper triangle of correlations
upper = app_train_corrs.where(np.triu(np.ones(app_train_corrs.shape),k=1).astype(np.bool))

#Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column]>threshold)]
print('There are %d columns to remove.' %(len(to_drop)))

#Drop Correlated Variables
app_train = app_train.drop(columns = to_drop)
app_test = app_test.drop(columns = to_drop)
print('Training shape:', app_train.shape)
#print(app_train.head())

# one-hot encoding of categorical variables
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)


train_labels = app_train['TARGET']

# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)

# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape:', app_train.shape)
print('Testing Featrues shape:', app_test.shape)

# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

from sklearn.preprocessing import MinMaxScaler, Imputer

# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)

from sklearn.ensemble import RandomForestClassifier

#Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)

#Train on the training data
random_forest.fit(train, train_labels)

#Extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature':features, 'importance':feature_importance_values})

#Make predicitions on the test data
predictions = random_forest.predict_proba(test)[:,1]

# Make a submission dataframe
submit = app_test[['SK_ID_CURR']]
submit['TARGET'] = predictions

# Save the submission dataframe
submit.to_csv('random_forest_baseline_domain.csv', index = False)
print('SAVED!!!')

'''
plt.figure(figsize = (8, 6))


# Heatmap of correlations
sns.heatmap(app_train_corrs, cmap = plt.cm.RdYlBu_r, vmin = 0.8, annot = True, vmax = 1.0)
plt.title('Correlation Heatmap');
plt.show()
'''
'''
# Save (to csv)
app_train_corrs.to_csv('app_train_corrs.csv', index = False)
'''
