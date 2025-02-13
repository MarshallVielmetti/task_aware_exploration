from matplotlib import pyplot as plt
import numpy as np




# represents an occupancy grid map
# USEAGE:
# draw shapes (rectangle, circle) on the map
# call init_ogm to initialize the ogm
class Map:
    def __init__(self, size):
        self.size = size
        
        # occupancy stores binary 0-1 values
        # 1 means the cell is free
        # 0 means the cell is occupied
        # 0-0 is the bottom left corner of the map
        # self.occupancy = [[1 for _ in range(size)] for _ in range(size)]
        self.occupancy = np.ones((self.size, self.size))

        # ogm
        # 1 means cells is known free
        # 0 means cells is known occupied
        # 0.5 means cells is unknown
        self.ogm = np.ones((self.size, self.size)) * 0.5


    # HELPER FUNCTIONS TO INITIALIZE THE MAP

    # add a rectangle to the occupancy grid
    # x, y is the bottom-left corner of the rectangle
    def add_rectangle(self, x, y, width, height):
        if (x + width > self.size) or (y + height > self.size):
            raise ValueError("Rectangle out of bounds")

        for i in range(x, x + width):
            for j in range(y, y + height):
                self.occupancy[j][i] = 0

    def add_circle(self, x, y, radius):
        if (x + radius > self.size) or (y + radius > self.size):
            raise ValueError("Circle out of bounds")

        for i in range(x - radius, x + radius):
            for j in range(y - radius, y + radius):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    self.occupancy[j][i] = 0


    # Plot just the obstacles
    def plot_obstacles(self, ax):
        ax.imshow(self.occupancy, cmap='gray', origin='lower')

    def plot_ogm(self, ax):
        ax.imshow(self.ogm, cmap='gray', vmin=0, vmax=1,origin='lower')

    def get_occupancy(self, x, y):
        return self.occupancy[y][x] # indexed as row, column
    
    # returns a np list containing the frontier cells
    def get_frontier(self):
        # iterate through the ogm
        # if a cell is unknown, check its neighbors
        # if a neighbor is known free, add the cell to the frontier
        # I'm using a definition of frontier cells as known cells that have an unknown neighbor...
        # don't want to navigate to unknown cells directly!
        frontier = []
        for y, x in np.ndindex(self.ogm.shape):
            if self.ogm[y][x] != 1: # only consider known free cells
                continue

            # check the neighbors
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if x + dx < 0 or x + dx >= self.size or y + dy < 0 or y + dy >= self.size:
                    continue

                # found a frontier cell -- neighbors are unexplored!
                if self.ogm[y + dy][x + dx] == 0.5:
                    frontier.append((x, y))
                    break

        return frontier


    # returns all valid neighbors of a point (row, col)
    def get_neighbors(self, point: tuple) -> list:
        y, x = point

        neighbors = [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]

        # remove out of bounds neighbors
        neighbors = [n for n in neighbors if n[0] >= 0 and n[0] < self.size and n[1] >= 0 and n[1] < self.size]

        return neighbors   



