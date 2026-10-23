######################################################

#  Student Name: Eddie Wenker

#  Course Code: CSCI 460 -- Fall 2026

#  Assignment Due Date: 10/25/26

#  GitHub Link: (https://github.com/Wenker-ed/CSCI-460-Programming-Assignment-I)

######################################################

import pandas as pd
# import performance metric libraries
from sklearn.metrics import (
accuracy_score,
confusion_matrix,
ConfusionMatrixDisplay,
f1_score,
classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

#Load in the data set

df = pd.read_csv('bank-full.csv', delimiter=';')
print(df.head()) #The data is related with direct marketing campaigns of a Portuguese banking institution. 
#The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required,
# in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. )

#Analyze the data set

print(df.shape) #data frame has 45211 rows and 17 columns (16 features + 1 target))
print(df.isnull().any()) #no nulls

#Map all categorical variables to integers

#age - numerical feature, no mapping needed
df['job'] = df['job'].map({"admin.": 0, "unknown": 1, "unemployed": 2, "management": 3, "housemaid": 4, "entrepreneur": 5, "student": 6, "blue-collar": 7, "self-employed": 8, "retired": 9, "technician": 10, "services": 11})
df['marital'] = df['marital'].map({"married": 0, "divorced": 1, "single": 2})
df['education'] = df['education'].map({"unknown": 0, "secondary": 1, "primary": 2, "tertiary": 3})
df['default'] = df['default'].map({'yes': 1, 'no': 0})
#balance - numerical feature, no mapping needed
df['housing'] = df['housing'].map({'yes': 1, 'no': 0})
df['loan'] = df['loan'].map({"yes": 1,"no": 0}) # map 1 or 0 on all rows of df[‘loan’]
  # related with the last contact of the current campaign:
df['contact'] = df['contact'].map({'unknown': 0, 'telephone': 1, 'cellular': 2})
#day - numerical feature, no mapping needed
df['month'] = df['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
#duration - numerical feature, no mapping needed
#Other attributes
#campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#pdays: number of days that passed by after the client was last contacted from a previous campaign 
# (numeric, -1 means client was not previously contacted)
#previous: number of contacts performed before this campaign and for this client (numeric)
df['poutcome'] = df['poutcome'].map({'unknown': 0, 'other': 1, 'failure': 2, 'success': 3})
df['y'] = df['y'].map({'yes': 1, 'no': 0})

print(df.head()) #display the first few rows of the dataframe after mapping)
print(df.isnull().any()) #mapping was successful if no null values are present)

#separate the features and target variable
x = df.drop(columns=['y'])
y = df['y']
print(x.shape)
print(y.shape)

## Task II: Train and Test a Naive Bayes Model

accuracy_saved = np.array([])
f1_saved = np.array([])
for i in range(10): # repeat the experiment 10 times
    #split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y)
    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy_saved = np.append(accuracy_saved, accuracy)
    f1_saved = np.append(f1_saved, f1)
    print(f"Iteration {i+1}: Accuracy = {accuracy}, F1 Score = {f1}")

print("Accuracy saved:", accuracy_saved)
print("F1 saved:", f1_saved)
print("Mean Accuracy:", np.mean(accuracy_saved))
print("Mean F1 Score:", np.mean(f1_saved))
print("Standard Deviation of Accuracy:", np.std(accuracy_saved))
print("Standard Deviation of F1 Score:", np.std(f1_saved))

## Task III: report impact of splitting the data

## Task III: report impact of splitting the data
# Repeat the previous experiment 9 more times to measure the performance of the Naive Bayes model
# Start with a training split of 10% and run 10 times
# then repeat after 10 iterations with a training split of 20%
accuracy_saved_all = []
f1_saved_all = []
for i in [.9, .8, .7, .6, .5, .4, .3, .2, .1]:
    accuracy_saved = []
    f1_saved = []

    for j in range(10): # repeat the experiment 10 times
        #split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=i, stratify=y) 
        #no random_state because that will give us the same split each time
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        accuracy_saved.append(accuracy)
        f1_saved.append(f1)
        train_size = 1 - i
        print(f"Training Size {train_size:.1f}: Iteration {j+1}: Accuracy = {accuracy}, F1 Score = {f1}")
    accuracy_saved_all.append(accuracy_saved)
    f1_saved_all.append(f1_saved)

accuracy_saved_all = np.array(accuracy_saved_all)
f1_saved_all = np.array(f1_saved_all)

for i in range(len(accuracy_saved_all)):
    print(f"{(i+1) * 10}% training split: accuracy mean = {np.mean(accuracy_saved_all[i])}, std = {np.std(accuracy_saved_all[i])}")
    print(f"{(i+1) * 10}% training split: f1 mean = {np.mean(f1_saved_all[i])}, std = {np.std(f1_saved_all[i])}")

plt.plot([np.mean(accuracy_saved_all[i]) for i in range(len(accuracy_saved_all))], label='Accuracy')
plt.plot([np.mean(f1_saved_all[i]) for i in range(len(f1_saved_all))], label='F1')
plt.xlabel('Training Split (%)')
plt.xticks(range(len(accuracy_saved_all)), [f"{(i+1) * 10}%" for i in range(len(accuracy_saved_all))])
plt.ylabel('Mean Value')
plt.title('Mean of Accuracy and F1 Score')
plt.legend()
plt.show()

plt.plot([np.std(accuracy_saved_all[i]) for i in range(len(accuracy_saved_all))], label='Accuracy')
plt.plot([np.std(f1_saved_all[i]) for i in range(len(f1_saved_all))], label='F1')
plt.xlabel('Training Split (%)')
plt.xticks(range(len(accuracy_saved_all)), [f"{(i+1) * 10}%" for i in range(len(accuracy_saved_all))])
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Accuracy and F1 Score')
plt.legend()
plt.show()