import matplotlib.pyplot as plt
import numpy as np

points = {'X':[],'Y':[]}
k = 0
with open("output.txt", "r") as f:
    X, Y = [], []
    for i in f.readlines():
        k += 1
        x, y = i.split(',')
        if x == 'end':
            points['X'].append(np.array(X))
            points['Y'].append(np.array(Y))
            X = []
            Y = []
        else:
            X.append(float(x))
            Y.append(float(y))
# print(*X)
# print()
# print(*Y)
mx = 0
for i in range(len(points['Y'])):
    mx = max(mx, (max(points['Y'][i])))
for i in range(len(points['Y'])):
    points['Y'][i] = mx - points['Y'][i]
for X, Y in zip(points['X'], points['Y']):
    plt.plot(X, Y)
plt.axis('equal')
plt.show()
