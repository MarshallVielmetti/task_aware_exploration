from map import Map


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np


def frontier_example():
    map = Map(100)

    # draw a square in the middle of the map
    # map.add_rectangle(40, 40, 20, 20)

    map.add_rectangle(20, 40, 20, 20)
    map.add_rectangle(60, 40, 20, 20)

    fig, ax = plt.subplots()

    # reveal bottom half of the map
    map.ogm[0:50, :] = map.occupancy[0:50, :]

    map.plot_ogm(ax)

    frontier = map.get_frontier()
    for f in frontier:
        ax.plot(f[0], f[1], 'yo')

    ax.set_title("Frontier Example")
    plt.savefig("Images/frontier_example.png")
    plt.show()
    
    fig, ax = plt.subplots()
    map.plot_ogm(ax)
    
    frontier = np.array(frontier)
    dbscan = DBSCAN(eps=10, min_samples=1)

    print("start")
    labels = dbscan.fit_predict(frontier)
    print(labels)

    unique_labels = set(labels)

    colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    for k in unique_labels:
        if k == -1:
            continue

        mask = labels == k
        cluster = frontier[mask]

        color = colors[k % len(colors)]
        ax.plot(cluster[:, 0], cluster[:, 1], c=color, marker='o')


    ax.set_title("Clustered Frontier Example")
    plt.savefig("Images/clustered_frontier_example.png")
    plt.show()


if __name__ == "__main__":
    frontier_example()