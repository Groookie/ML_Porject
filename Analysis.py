import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn.preprocessing import robust_scale
from sklearn.neighbors import LocalOutlierFactor
import warnings
# suppress warning
warnings.simplefilter("ignore")
data = pd.read_csv("./data.csv", sep=';')
t = data.isna()
if data.isna().values.any():
    print("Contain null value")
else:
    print("No null value")

# correlation
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
f = plot.figure(figsize=(10, 8))
plot.matshow(x.corr(), fignum=f.number)
plot.xticks(range(x.shape[1]), x.columns, rotation=45)
plot.yticks(range(x.shape[1]), x.columns)
cb = plot.colorbar()
cb.ax.tick_params(labelsize=14)
plot.show()

# normalize
normalized_x = robust_scale(x)

# detect outlier
detector = LocalOutlierFactor(n_neighbors=10)
select = detector.fit_predict(normalized_x)
labels = y[select == 1]
value = np.bincount(labels)
labels = np.arange(len(value))


