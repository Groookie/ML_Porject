import numpy as np
from sklearn.manifold import TSNE
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
from mpl_toolkits.mplot3d import Axes3D

# suppress warning
warnings.simplefilter("ignore")
# load and preprocess data
data = pd.read_csv("./data.csv", sep=';')
x = data.iloc[:, :-1]
y = data.iloc[:, -1].to_numpy()
x = robust_scale(x)
data_num = len(y)
select = np.random.choice(data_num, data_num // 5)
train_x = x[np.delete(np.arange(data_num), select)]
train_y = y[np.delete(np.arange(data_num), select)]
test_x = x[select]
test_y = y[select]

# use multiple methods
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

x_visualise = TSNE(n_components=3).fit_transform(test_x)

colors = np.arange(10)
# visualise
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
unique_label = np.unique(predict_y)
for label in unique_label:
    t = x_visualise[predict_y == label]
    ax.scatter(t[:, 0], t[:, 1], t[:, 2], marker='o')
plt.show()
