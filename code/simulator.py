from map import Map
from agent import Agent
from algorithm import Algorithm

import matplotlib.pyplot as plt


class Simulator:
    def __init__(self, env: Map, agents: list[Agent], alg: Algorithm):
        self.env = env
        self.agents = agents
        self.alg = alg

    def update(self):
        # each agent senses the environment
        for agent in self.agents:
            agent.sense(self.env)

        # update the agents
        self.alg.update_agents(self.env, self.agents)


    def update_plot(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')

        ax[0].set_xlim(0, self.env.size)
        ax[0].set_ylim(0, self.env.size)

        ax[1].set_xlim(0, self.env.size)
        ax[1].set_ylim(0, self.env.size)


        # begin by plotting the obstacles
        self.env.plot_obstacles(ax[0])
        ax[0].set_title("Obstacles")

        # now plot the current OGM
        self.env.plot_ogm(ax[1])
        ax[1].set_title("OGM")


        # now overlay the agents on both the OGM and the obstacles
        for agent in self.agents:
            agent.plot(ax[0])
            agent.plot(ax[1])




    
