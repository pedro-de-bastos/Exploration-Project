import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.svm import SVR



x = np.linspace(0, 100, num=1000)
y = np.cos(x) + 0.1*sts.norm.rvs(0, 1, size=len(x))

regr = SVR()
regr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
preds = regr.predict(x.reshape(-1, 1))
plt.plot(x, y)
plt.scatter(x, preds)

plt.show()