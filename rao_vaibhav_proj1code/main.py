#import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import os 
#print("DEBUG:::: Listing dataset file DIR:", os.listdir("./dataset_files"))

#No Headers in the csv so ignore set header=None so that row not ignored
dataset = pd.read_csv("./dataset_files/wdbc.dataset", header=None)

#print("DEBUG:: wdbc.dataset file has {0[0]} rows and {0[1]} columns.".format(dataset.shape))
#dataset.head()
#dataset.info()

X = dataset.loc[:, 2:].values # 30 Features in csv starting from Column 2 till last
Y = dataset.loc[:, 1].values # Target variable (DIAGNOSIS) which is column 1.

#Encoding any categorical values
#Changes M and B to 1 and 0, respectively using the label encoder on Y
from sklearn.preprocessing import LabelEncoder
le_Y = LabelEncoder()
Y = le_Y.fit_transform(Y)
#print('DEBUG:: Y after label encode::',Y)

#Splitting the dataset into Train and Test. train_test_split will select the data randomly!
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
#As per the problem statement,we use 20% of the dataset for testing purposes and 80% will be for training our model

# As per requirement shuffle the data. 
# NOT REQUIRED Since we are already selecting random data
## WOULD CAUSE DIFFERENT RESULTS EVERYTIME!!!
#shuffle_index = np.random.permutation(len(X_train))
#X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]

#Feature Scaling the Data Set so that it is within a magnitude scale
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, precision_score, recall_score

sgd_clf = SGDClassifier(max_iter=33, random_state=42) #Achieved concurency at 33
sgd_clf.fit(X_train, Y_train)
prediction = sgd_clf.predict(X_test)

confusionmatrix = confusion_matrix(Y_test, prediction)
print('Confusion Matrix for SGD Classifier.predict::')
print(confusionmatrix)
print ("Accuracy Score for SGD_CLF.PREDICT:", accuracy_score(prediction, Y_test))
print ("Precision Score for SGD_CLF.PREDICT:", precision_score(Y_test, prediction))
print ("Recall Score for SGD_CLF.PREDICT:", recall_score(Y_test, prediction))

#### Let's also see accuracy for different subset of the data to make sure we are on track!!!
score = cross_val_score(sgd_clf, X_train, Y_train, cv=3, scoring="accuracy")
print ("Cross Value Score Accuraccy:: ", score)
#Y_test_pred = cross_val_predict(sgd_clf, X_test, Y_test, cv=3)
Y_train_pred = cross_val_predict(sgd_clf, X_train, Y_train, cv=3)
print('Confusion Matrix for Cross_val_predict for training data::')
print (confusion_matrix(Y_train, Y_train_pred))
print ("Accuracy Score for Cross_Val_Predict:", accuracy_score(Y_train_pred, Y_train))
print ("Precision Score for Cross_Val_Predict:", precision_score(Y_train, Y_train_pred))
print ("Recall Score for Cross_Val_Predict:", recall_score(Y_train, Y_train_pred))

#print('DEBUG:::: Confusion Matrix for Perfect Prediction::')
#Y_train_perfect_predictions = Y_test
#print (confusion_matrix(Y_test, Y_train_perfect_predictions))

#### Cross val predict of training data
y_scores = cross_val_predict(sgd_clf, X_train, Y_train, cv=3, method="decision_function")

#precisions_training, recalls_training, thresholds_training = precision_recall_curve(Y_train, y_scores)
#precisions_test, recalls_test, thresholds_test = precision_recall_curve(Y_test, prediction)

"""
Plotting the ROC Curves for training and test data. 
Taking Malignant (1) as the True Positive case.
Benign is class 0 as True Negative Case.
"""

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(Y_train, y_scores)

fpr_test,tpr_test,threshold_test = roc_curve(Y_test, prediction)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(".", "plotted_curves", fig_id + ".png") #join various path components
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
save_fig("roc_curve_plot_training")
plt.show()

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr_test, tpr_test)
save_fig("roc_curve_plot_test")
plt.show()

