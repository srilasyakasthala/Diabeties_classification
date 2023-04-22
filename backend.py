import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import pickle

# Load the dataset
df = pd.read_csv('ML Pro\diabetes.csv')
df = df.head(700)

# Display the first few rows of the dataset
df.head()

# Check the shape of the dataset
print("Dataset shape:", df.shape)

# Split the dataset into features (x) and target (y) variables
x = df.iloc[:, [1,2]]
y = df.iloc[:, -1]

# Display the first few rows of x and y
x.head()
y.head()

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=10)

# Print the shapes of training and test data
print("Training data shape: ", x_train.shape)
print("Test data shape: ", x_test.shape)

# Scale the features using StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Train a Support Vector Machine (SVM) classifier with linear kernel
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)

# Predict the target variable on the test data
y_pred = classifier.predict(x_test)

# Calculate accuracy score for the linear kernel SVM
print('Accuracy Score with linear kernel:', metrics.accuracy_score(y_test, y_pred))



pickle.dump(classifier,open('model.pkl','wb'))