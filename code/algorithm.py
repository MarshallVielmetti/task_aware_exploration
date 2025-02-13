from map import Map
from agent import Agent

import numpy as np

class Algorithm:
    # assumes agents have already sensed
    def update_agents(self, map: Map, agents: list[Agent]):
        pass


class RandomWalk(Algorithm):
    def update_agents(self, map: Map, agents: list[Agent]):
        # for now, just move the agents randomly
        for agent in agents:
            delta = np.random.rand(2) - 0.5
            agent.update(map, *delta)



class SimpleFrontierBased(Algorithm):
    needs_to_plan: bool = True
    
    # stores a map from each agent to the path to their objective
    agent_plans: dict[Agent, list[tuple[int, int]]] = {}

    def update_agents(self, map: Map, agents: list[Agent]):
        if self.needs_to_plan:
            self.plan(map, agents)

    def plan(self, map, agents):
        target_locations = self.calc_optimal_targets(map, agents)
        for agent, target in zip(agents, target_locations):
            self.agent_plans[agent] = self.a_star(map, agent, target)

    def calc_optimal_targets(self, map, agents):
        pass

    def a_star(self, map, agent, target):
        pass

        
