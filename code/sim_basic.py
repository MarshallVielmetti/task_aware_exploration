from algorithm import RandomWalk
from map import Map
from agent import Agent

from simulator import Simulator

from matplotlib import pyplot as plt

def init_basic_map():
    map = Map(1000)

    map.add_rectangle(100, 100, 200, 300)

    map.add_circle(500, 500, 100)

    return map


def sim_basic():
    basic_map = init_basic_map()

    agents = []
    agents.append(Agent(100, 100, 20, 100))
    agents.append(Agent(200, 100, 20, 100))
    agents.append(Agent(300, 100, 20, 100))

    alg = RandomWalk()

    sim = Simulator(basic_map, agents, alg)

    sim.update_plot()
    plt.show()

    for i in range(100):
        sim.update()

    sim.update_plot()
    plt.show()

    for i in range(100):
        sim.update()

    sim.update_plot()
    plt.show()

    for i in range(100):
        sim.update()

    sim.update_plot()
    plt.show()


if __name__ =="__main__":
    sim_basic()