import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### Basic Graphs


X = np.array([1, 2, 3, 4, 16])
Y = np.array([2, 6, 8, 14, 32])

x = np.arange(5, 15) ## create linear array
y = x ** 2 ## create quadratic array

plt.plot(x, y, label="y = x^2", color="green", linestyle="--", marker="o", markersize=10, markerfacecolor="blue", markeredgecolor="red")
plt.plot(X, Y, label="line 2", color="red", linestyle="-", marker="o", markersize=10, markerfacecolor="red")

plt.title("Basic Graph", fontdict={"fontname": "Comic Sans MS"}, fontsize=20)

plt.xlabel("X-axis", fontsize=20)
plt.ylabel("Y-axis", fontsize=20)

plt.legend()
plt.show()

