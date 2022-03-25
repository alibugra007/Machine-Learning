import matplotlib.pyplot as plt
import numpy as np
import math

variables = np.random.uniform(0, 100, [200, 2])
# variables with two features

amount_of_centroids = int(input("Amount of Centroids:"))
# We're giving the amount of centroids which is basically how much clustering we are doing.

centroids = np.random.uniform(0, 100, [amount_of_centroids, 2])
# centroids features.

b = np.zeros((variables.shape[0], centroids.shape[0]))
# information about which variable in which cluster.


plt.ion()
plt.figure(figsize=(6, 6))

for iter in range(0, 4):  # iteration of how many times centroids move.

    plt.cla()  # clear plot
    for i in range(0, variables.shape[0]):  # we'll temporarily assign variables to b.
        # and then we'll assign 0 to the values in row i and then 1 to the smallest.
        for j in range(0, centroids.shape[0]):
            b[i][j] = math.dist(variables[i], centroids[j])

        # we will make the smallest 0 and the others 1.
        min_value = np.amin(b[i])
        min_value_index = np.where(min_value == b[i])
        b[i] = 0
        b[i][min_value_index] = 1

    for i in range(0, b.shape[-1]):  # We've taken centroids one time.
        x_points = (np.dot(variables[:, 0], b[:, i]) / np.sum(b[:, i], axis=0))
        y_points = (np.dot(variables[:, 1], b[:, i]) / np.sum(b[:, i], axis=0))
        centroids[i] = [x_points, y_points]

    for i in range(0, variables.shape[0]):  # Trying to draw connection lines.
        min_value_index = np.where(b[i] == 1)
        x_points = [centroids[int(min_value_index[0])][0], variables[i][0]]
        y_points = [centroids[int(min_value_index[0])][1], variables[i][1]]

        plt.plot(x_points, y_points)  # First we're plotting lines.
        # and then plot variables and centroids. This way, we prevented conflicts.

    plt.plot(variables[:, 0], variables[:, 1], "o")  # variables
    plt.plot(centroids[:, 0], centroids[:, 1], "s", color="red")  # centroids

    plt.pause(2)


plt.show()

