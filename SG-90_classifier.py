# Importing the required libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

warnings.filterwarnings('ignore')

test_file_path = input('Enter test file path: ')    # Getting the test file

train_data = pd.read_csv('penguins_train.csv')  # Reading the training data
test_data = pd.read_csv(test_file_path) # Reading the test data 

# Imputing the data to compensate for the missing values
imputer = SimpleImputer(strategy='most_frequent')
train_imputer = imputer.fit_transform(train_data.copy())
tr_data = pd.DataFrame(train_imputer, columns=list(train_data.columns))
test_imputer = imputer.fit_transform(test_data.copy())
ts_data = pd.DataFrame(test_imputer, columns=list(test_data.columns))

features = list(tr_data.drop('Species', axis=1).columns)

# Feature Engineering of the categorical data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Encoding the training data into usable format
df = tr_data.copy()
string_columns = df[['Island','Clutch Completion', 'Sex', 'Species']]
encoded_data = {c:LabelEncoder() for c in string_columns}
for cols, attributes in encoded_data.items():
    df[cols] = encoded_data[cols].fit_transform(df[cols])

# Encoding the test data into a usable format
test_df = ts_data.copy()
str_columns = test_df[['Island','Clutch Completion', 'Sex']]
enc_data = {c:LabelEncoder() for c in str_columns}
for cl, att in enc_data.items():
    test_df[cl] = enc_data[cl].fit_transform(test_df[cl])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

X_train = df[features]  # Specifying the training data features
y_train = df['Species'] # Specifying the training labels

X_test = test_df.copy() # Copying the test data to X_test variable


# Function to create trees
def create_trees(X_train, Y_train, max_depth, min_size, n_features, criterion):
    D1 = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split = min_size, max_features=n_features)
    D1.fit(X_train, Y_train)
    return D1


# Building an all-vs-all Random-Forest multiclass classifier 
def Random_forest(X_train, X_test, max_depth, min_size, estimators, min_features):
  
  # Initialising parameters to be selected randomly for tree formation
  max_depth_list = list(range(2,max_depth))
  criterion_list = ["gini", "entropy"]
  min_size_list = list(range(3,min_size))
  n_features_list = list(range(min_features ,X_train.shape[1]))
  max_rows = X_train.shape[0]

  all_preds = []

  for i in tqdm(range(estimators), desc='Running Classifier, please wait'):     # creating 'n' decision trees | n = estimators
    ids = np.random.randint(max_rows, size=(int(.8*max_rows)))
    X_train_split = []
    Y_train_split = []
    for x in ids:     # Bootstrapping to compensate for less data
      row = X_train.sample()
      X_train_split.append(list(row.values[0]))
      Y_train_split.append(y_train.iloc[row.index[0]])

    X_train_split = np.array(X_train_split)
    Y_train_split = np.array(Y_train_split)

    # Choosing the random parameters 
    id_max_depth = np.random.randint(len(max_depth_list),size=1)[0]
    id_criterion = np.random.randint(len(criterion_list),size=1)[0]
    id_min_size = np.random.randint(len(min_size_list),size=1)[0]
    id_feature = np.random.randint(len(n_features_list),size=1)[0]

    tree = create_trees(X_train_split, Y_train_split, max_depth_list[id_max_depth], min_size_list[id_min_size],
                        n_features_list[id_feature], criterion = criterion_list[id_criterion])
    pred = tree.predict(X_test)
    all_preds.append(pred)    # Adding all prediction to an array

  all_preds = np.array(all_preds)

  Y_pred = []   # Initialising final prediction
  Y_prob = []     # Initialising class prediction probability
  for i in range(all_preds.shape[1]):
    temp = all_preds[:,i]
    counts = np.bincount(temp)

    # Storing class probability of prediction
    if len(counts) == 1:
      cnt = [counts[0]/estimators, 0, 0]
      Y_prob.append(cnt)
    elif len(counts) == 2:
      cnt = [counts[0]/estimators, counts[1]/estimators, 0]
      Y_prob.append(cnt)
    else:
      Y_prob.append([counts[0]/estimators, counts[1]/estimators, counts[2]/estimators])

    Y_pred.append(np.argmax(counts))    # Storing the most voted prediction

  Y_pred = np.array(Y_pred)
  Y_prob = np.array(Y_prob)

  return Y_pred, Y_prob

# Getting predictions
Y_pred, Y_prob = Random_forest(X_train, X_test, max_depth=8, min_size=4,
                        estimators=150, min_features=4)

# Converting predictions to the original labels
predictions = encoded_data['Species'].inverse_transform(Y_pred)

# Saving the predictions in 'predictions.csv' file
pd.DataFrame(data=predictions, columns=['predictions']).to_csv('predictions.csv')
print('\nRuntime executed successfully\n')
print("Predictions are stored in 'predictions.csv' file in the current working directory")