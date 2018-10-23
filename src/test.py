# importing necessary libraries
import pandas as pd
import os
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

#loading the data
data_dir ="/Users/vishadnehal/Desktop/EE542/Lab10/vishad/data/"

data_file = data_dir + "miRNA_matrix.csv"

df = pd.read_csv(data_file)
print("Data has been loaded")
df.label = pd.factorize(df.label)[0]
df.pop('file_id')
df["label"]
# all null rows in test
test = df[df["label"].isnull()]
test.pop('label')
# all non null rows in train
train = df[df["label"].notnull()]

  
# dividing X, y into train and test data 
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,0:train.shape[1]-1], train["label"], test_size=0.5647, random_state=0)
df = pd.concat([X_train,y_train],axis=1)
y = df.label
X = df.drop("label", axis=1)
 
# Train model using Logistic Regression
clf_1 = LogisticRegression(penalty='l2',multi_class='multinomial',solver = 'newton-cg',max_iter=500).fit(X, y)

#Predict on training set
pred_y_1 = clf_1.predict(X)
print("train accuracy",accuracy_score(y, pred_y_1) )
#Predict on testing set
y_test_predicted = clf_1.predict(X_test)
print("test accuracy",accuracy_score(y_test, y_test_predicted) )

def specificity_score(y_true, y_predict):
    '''
    true_negative rate
    '''
    true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
    real_negative = len(y_true) - sum(y_true)
    return true_negative / real_negative 

precision = precision_score(y_test, y_test_predicted, average='micro')
accuracy = accuracy_score(y_test, y_test_predicted)
f1 = f1_score(y_test, y_test_predicted, average='micro')
recall = recall_score(y_test, y_test_predicted, average='micro')
specificity = specificity_score(y_test, y_test_predicted)

precisions = [precision]
accuracies =[accuracy]
f1_scores = [f1]
recalls = [recall]
specificities = [specificity]


s = pd.Series(
    [precision, accuracy, f1, recall, specificity],
    index = ["Precision", "Accuracy", "F1", "Recall", "Specificity"]
)

#Set descriptions:
plt.xlabel('Logistic Regression')

#Set tick colors:
ax = plt.gca()
ax.tick_params(axis='x', colors='black')
ax.tick_params(axis='y', colors='black')

#Plot the data:
my_colors = ['r', 'b', 'k', 'c']  # red, green, blue, black, etc.

s.plot( 
    kind='bar', 
    color=my_colors,
)


x = StandardScaler().fit_transform(train.iloc[:,0:train.shape[1]-1])

pca = PCA(n_components=3)
pca_result = pca.fit_transform(x)
pca_result = pd.DataFrame(pca_result)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
df['pca-one'] = pca_result.iloc[:,0]
df['pca-two'] = pca_result.iloc[:,1]
df['pca-three'] = pca_result.iloc[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

import seaborn as sns
sns.pairplot(x_vars=["pca-one"], y_vars=["pca-two"], data=df, 
hue="label", height=5)

sns.pairplot(x_vars=["pca-three"], y_vars=["pca-two"], data=df, 
hue="label", height=5)

sns.pairplot(x_vars=["pca-one"], y_vars=["pca-three"], data=df, 
hue="label", height=5)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,0:train.shape[1]-1], train["label"], test_size=0.5647, random_state=0)

d = pd.DataFrame(0, index=np.arange(len(y_test)), columns= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43])
for i in range(0,len(y_test)):
    d.iloc[i][y_test.values[i]] = 1

mclass = LogisticRegression(C=50. ,
                         multi_class="multinomial",
                         penalty="l1", solver="saga", tol=0.1)

clf = OneVsRestClassifier(mclass)
y_score = clf.fit(X_train, y_train).decision_function(X_test)

n_classes = 40

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(d.values[:,i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure()

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    #plt.legend(loc="top right")
plt.show()
########################################## training a KNN classifier #######################################
#knn = KNeighborsClassifier(n_neighbors = 7)
#knn.fit(X,y)
#knn = KNeighborsClassifier(n_neighbors = 7).fit(X,y)
# accuracy on X_test 
#trainaccuracy = knn.score(X,y) 
#print("training accuracy",trainaccuracy)
#test accuracy
#testaccuracy = knn.score(X_test,y_test)
#print("test accuracy",testaccuracy)