import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import robust_scale
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode
import warnings
# suppress warning
warnings.simplefilter("ignore")
data = pd.read_csv("./data.csv", sep=';')
x = data.iloc[:, :-1]
y = data.iloc[:, -1].to_numpy()

# normalize
x = robust_scale(x)
data_num = len(y)

# part1 predict density
density = x[:, 7]
regression_x = np.delete(x, 7, 1)
linearRegression = LinearRegression()

select = np.random.choice(data_num, data_num // 5)
train_x = regression_x[np.delete(range(data_num), select)]
train_y = density[np.delete(range(data_num), select)]
test_x = regression_x[select]
test_y = density[select]
linearRegression.fit(train_x, train_y)
predict_y = linearRegression.predict(test_x)
print("l2 distance:" + str(np.linalg.norm(test_y - predict_y)))
# part2 use knn to predict quality
classifier = KNeighborsClassifier()
# Try other classifier if you want
# classifier = MLPClassifier()
# classifier = SVC()
# classifier = DecisionTreeClassifier()

select = np.random.choice(data_num, data_num // 5)
train_x = x[np.delete(np.arange(data_num), select)]
train_y = y[np.delete(np.arange(data_num), select)]
test_x = x[select]
test_y = y[select]
classifier.fit(train_x, train_y)
predict_y = classifier.predict(test_x)
print("Knn Prediction Accuracy: {}".format(accuracy_score(test_y, predict_y)))
# Use confusion matrix to analyze the reason
cm = confusion_matrix(test_y, predict_y)
classes = np.arange(0, 10)
t = unique_labels(test_y, predict_y)
classes = classes[t]
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=classes, yticklabels=classes,
       title="confusion matrix",
       ylabel='True label',
       xlabel='Predicted label')
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()
# combine multiple methods
classifier1 = KNeighborsClassifier()
classifier2 = MLPClassifier()
classifier3 = SVC()
classifier4 = DecisionTreeClassifier()
classifier1.fit(train_x, train_y)
classifier2.fit(train_x, train_y)
classifier3.fit(train_x, train_y)
classifier4.fit(train_x, train_y)
predict_y1 = classifier1.predict(test_x)
predict_y2 = classifier2.predict(test_x)
predict_y3 = classifier3.predict(test_x)
predict_y4 = classifier4.predict(test_x)
# combine result
predict_y = np.stack([predict_y1, predict_y2, predict_y3, predict_y4])
predict_y, _ = mode(predict_y)
predict_y = predict_y.squeeze()
print("Combined method Prediction Accuracy: {}".format(accuracy_score(test_y, predict_y)))
