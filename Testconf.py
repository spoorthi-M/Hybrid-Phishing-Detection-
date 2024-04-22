
from sklearn.metrics import confusionmatrix
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

arr_random = np.random.randint(low=0, high=2, size=(1000,4))
testdataframe=pd.DataFrame(arr_random, columns=["A","B","C","target"])
print(type(testdataframe))
ddataset=testdataframe
a="NB"
try:
    nb = GaussianNB()

    nb.fit(X_train,Y_train)

    Y_pred_nb = nb.predict(X_test)
except:
    nb = GaussianNB()
    
    
nbscore=confusionmatrix.accuracy_score()
labels=[0,1]
X, Y = make_classification(n_samples=500, n_classes=2, n_features=20, random_state=10)
data=confusionmatrix.heatmap(X,labels)
actual,predicted=confusionmatrix.generatelabels(ddataset,X,labels,a)
len(actual)
# confusion matrix
matrix = confusion_matrix(actual,predicted)
print('Confusion matrix : \n',matrix)

matrix = classification_report(actual,predicted)
print('Classification report : \n',matrix)



from sklearn import metrics
confusion_matrix1 = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1, display_labels = labels)
cm_display.plot()
plt.savefig("Nbcon.png")
plt.show()

a="LR"
try:
    nb = LogisticRegression()

    nb.fit(X_train,Y_train)

    Y_pred_nb = nb.predict(X_test)
except:
    nb = LogisticRegression()
    
    
nbscore=confusionmatrix.accuracy_score()
#Confusion Matrix
labels=[0,1]
X, Y = make_classification(n_samples=500, n_classes=2, n_features=20, random_state=10)
data=confusionmatrix.heatmap(X,labels)
actual,predicted=confusionmatrix.generatelabels(ddataset,X,labels,a)
len(actual)
# confusion matrix
matrix = confusion_matrix(actual,predicted)
print('Confusion matrix : \n',matrix)

matrix = classification_report(actual,predicted)
print('Classification report : \n',matrix)



from sklearn import metrics
confusion_matrix1 = metrics.confusion_matrix(actual, predicted)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix1, display_labels = labels)
cm_display.plot()
plt.savefig("lrcon.png")
plt.show()
