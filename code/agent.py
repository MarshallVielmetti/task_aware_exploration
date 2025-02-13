from map import Map

import numpy as np
import matplotlib.patches as patches


# represents an agent in the environment
# these are basically just data classes which are manipualted by the agent's managing algorithm
# this is because currently I am using a centralized appraoch
class Agent:
    def __init__(self, x, y, sensing_radius, comms_radius):
        self.x = x
        self.y = y
        self.sensing_radius = sensing_radius
        self.comms_radius = comms_radius


    # update environment to reflect current sensor readings
    def sense(self, map: Map):
        # update the OGM to reflect the sensing radius
        # this paper assumes perfect sensing

        # cast out 60 rays -- may need to adjust this if necessary
        for i in range(0, 360):
            self.raytrace(map, i * np.pi / 180)


    # generally-- assume dx and dy are -1 or 1
    def update(self, map: Map, dx, dy):
        # verify bounds before updating
        if self.x + dx < 0 or self.x + dx >= map.size or self.y + dy < 0 or self.y + dy >= map.size:
            return
        
        # veirfy that the agent is not moving into an obstacle
        if map.occupancy[int(self.y + dy)][int(self.x + dx)] == 0:
            return

        self.x += dx
        self.y += dy


    # walk the ray and update the OGM
    def raytrace(self, map: Map, angle):
        # get the x and y components of the ray
        curr_pos = (self.x, self.y)
        ray_dir = (self.sensing_radius * np.cos(angle), self.sensing_radius * np.sin(angle))

        # normalize 
        ray_dir = ray_dir / np.linalg.norm(ray_dir) 

        # walk the ray
        for i in range(0, self.sensing_radius):
            curr_pos = curr_pos + ray_dir

            # check if the ray is out of bounds
            if curr_pos[0] < 0 or curr_pos[0] >= map.size or curr_pos[1] < 0 or curr_pos[1] >= map.size:
                break

            # check if the ray is hitting an obstacle -- a zero in the occupancy map
            if map.get_occupancy(int(curr_pos[0]), int(curr_pos[1])) == 0:
                break

            # update the OGM -- 1 means perfect information
            map.ogm[int(curr_pos[1])][int(curr_pos[0])] = 1


    def plot(self, ax):
        # plot the agent as a red circle
        ax.plot(self.x, self.y, 'ro', markersize=1)

        # plot the sensing radius as a dashed red circle
        sense_circ = patches.Circle((self.x, self.y), self.sensing_radius, color='r', fill=False)

        # plot the communication radius as a dashed blue circle
        comms_circ = patches.Circle((self.x, self.y), self.comms_radius, color='b', fill=False)

        ax.add_patch(sense_circ)
        ax.add_patch(comms_circ)





