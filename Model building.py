import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

dataset = pd.read_csv('new_appdata11.csv')
# print(dataset.head())

response = dataset['enrolled']
dataset = dataset.drop(columns='enrolled')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, response,
                                                    test_size=0.2,
                                                    random_state=0)

# Storing the user ids because we wont need them for the model
train_identifier = X_train['user']
X_train = X_train.drop(columns=['user'])
test_identifier = X_test['user']
X_test = X_test.drop(columns=['user'])

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X  = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.fit_transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

#Model building

from sklearn.linear_model import LogisticRegression as lr
classifier = lr(random_state=0, penalty='l1')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,recall_score,precision_score
cm = confusion_matrix(y_test,y_pred)

accuracy = accuracy_score(y_test,y_pred)
re_score = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# print(accuracy)
# print(re_score)
# print(precision)
# print(type(accuracy))

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Formatting Final Results
final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
final_results['predictions'] = y_pred
final_results = final_results[['user', 'enrolled', 'predictions']].reset_index(drop=True)
print(final_results)






















