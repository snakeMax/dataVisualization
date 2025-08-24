import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### Basic Graphs

x = np.array([1, 2, 3, 4, 16])
y = np.array([1, 4, 9, 16, 50])


plt.plot(x, y)

plt.title("Basic Graph", fontdict={"fontname": "Comic Sans MS"}, fontsize=20)

plt.xlabel("X-axis", fontsize=20)
plt.ylabel("Y-axis", fontsize=20)
plt.show()

