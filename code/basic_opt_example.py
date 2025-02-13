from agent import Agent
from map import Map
from simulator import Simulator

import matplotlib.pyplot as plt
import numpy as np
import pulp

import heapq


COMMS_RADIUS = 20


# perform djikstras algorithm to find the length of the shortest path
# from start to every other **known free** point on the map
def djikstra(map: Map, start: tuple):
    # priority queue of (distance, point) tuples

    x, y = start
    start = (y, x) #row, col

    pq = [(0, start)]

    cost = np.full(map.ogm.shape, np.inf)
    cost[start] = 0

    while pq:
        dist, point = heapq.heappop(pq)
        
        # infinite distance means no path
        if (dist == np.inf):
            break

        new_dist = dist + 1 # uniform cost of 1 to move between cells

        for neighbor in map.get_neighbors(point):
            # only add neighbors that are known free
            if map.ogm[neighbor] != 1: # occupied so skip
                continue

            if new_dist < cost[neighbor]:
                cost[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))


    return cost


def distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def calc_optimal_assignments(sim):
    # get the cost map for all the agents
    cost_maps = []
    for agent in sim.agents:
        print("djikstras for agent", agent.x, agent.y)
        cost_maps.append(djikstra(sim.env, (agent.x, agent.y)))

    cost_maps = np.array(cost_maps).transpose(1, 2, 0)

    print("shape of cost maps:", cost_maps.shape)

    cost_maps = np.array(cost_maps).reshape(100, 100, 3)
    plt.imshow(cost_maps[:, :, 0], cmap='gray', origin='lower')
    plt.show()

    # calculate the information gain for each cell
    # this is going to basically just be a large number if the cell is a frontier cell,
    # and 0 otherwise
    frontier = sim.env.get_frontier()
    info_gain = np.zeros(sim.env.ogm.shape)
    for cell in frontier:
        info_gain[cell[1]][cell[0]] = 1

    info_gain *= 100

    np.set_printoptions(threshold=np.inf)

    # now we assign the enforced edge set which imposes the active communication constraints
    edge_set = [(0, 1), (1, 2)] # agent 1 is like root of tree

    original_coords = [[(i, j) for i in range(sim.env.ogm.shape[0])] for j in range(sim.env.ogm.shape[1])]
    # print('shape of original coords', np.array(original_coords).shape)

    # force everything into correct shapes
    original_coords = np.array(original_coords).reshape(100, 100, 2)
    info_gain = info_gain.reshape(100, 100, 1)
    print('shape of original coords', original_coords.shape)
    print("shape of info gain", info_gain.shape)
    print("shape of cost maps", cost_maps.shape)
    print("shape of ogm", sim.env.ogm.shape)

    info_gain = info_gain.reshape(100, 100, 1)

    # stack the ogm, then the info gain main, then the cost maps
    # input = np.stack([sim.env.ogm, info_gain, original_coords] + cost_maps)
    input = np.concatenate((sim.env.ogm.reshape(100, 100, 1), info_gain, original_coords, cost_maps), axis=2)
    print("shape of input", input.shape)

    # input becomes a nxnx4+agents array:
    # 1. ogm
    # 2. info gain
    # 3. original coordinates (x)
    # 4. original coordinates (y)
    # 5. cost map for agent 0
    # ...


    # now flatten the input to a 1D array (of 5 tuples), taking only known free cells 
    free_cells = input[:, :, 0] == 1
    flattened = input[free_cells, :]

    print("shape of flattened", flattened.shape)

    flat_ogm = flattened[:, 0]
    flat_info_gain = flattened[:, 1]
    flat_orig_coords = flattened[:, 2:4]
    flat_cost_maps = flattened[:, 4:]


    NUM_AGENTS = len(sim.agents)


    # now we construct the optimization problem
    prob = pulp.LpProblem("OptimalAssignment", pulp.LpMaximize)

    # define variables -- a_ij is 1 if agent i is assigned to cell j
    a = [[pulp.LpVariable(f'a_{i}{j}', 0, 1, pulp.LpInteger) for j in range(flat_info_gain.shape[0])] for i in range(NUM_AGENTS)]

    # print("shape of a", np.array(a).shape)

    # for i in range(NUM_AGENTS):
    #     for j in range(flat_info_gain.shape[0]):
    #         print(flat_info_gain[j], flat_cost_maps[j, i])

    # define objective
    prob += pulp.lpSum((a[i][j]*(flat_info_gain[j] - flat_cost_maps[j, i]))for j in range(flat_info_gain.shape[0]) for i in range(NUM_AGENTS)), "optimal_assignment"


    # define constraints
    
    # each cell is assigned to at most one agent
    for j in range(flat_info_gain.shape[0]):
        prob += pulp.lpSum(a[i][j] for i in range(NUM_AGENTS)) <= 1, f"cell_{j}_single_assignment"

    # each agent is assigned to as most one cell
    for i in range(NUM_AGENTS):
        prob += pulp.lpSum(a[i][j] for j in range(flat_info_gain.shape[0])) == 1, f"agent_{i}_mandatory_assignment"

    # enforce the edge set
    for i, j in edge_set:
        for k in range(flat_info_gain.shape[0]):
            for l in range(flat_info_gain.shape[0]):
                # prob += (a[i][k]*a[j][l])*(euclid_squared()) <= COMMS_RADIUS, f"edge_set_({i}{j})_-_a_{i}{k}_and_a_{j}{l}"
                prob += pulp.LpConstraint(a[i][k] + a[j][l] - 1 <= distance_squared(flat_orig_coords[k], flat_orig_coords[l]) < COMMS_RADIUS**2, f"edge_set_({i}{j})_-_a_{i}{k}_and_a_{j}{l}")


    # solve the problem
    prob.solve()

    print(f"Status: {pulp.LpStatus[prob.status]}")
    print(f"Objective: {pulp.value(prob.objective)}")
    print("assignments:")
    optimal_assignments = []
    for i in range(NUM_AGENTS):
        for j in range(free_cells.shape[0]):
            if a[i][j].varValue == 1:
                print(f"Agent {i} assigned to cell {j}")
                optimal_assignments.append((i, j))


    return optimal_assignments

def basic_opt_example():
    map = Map(100)

    # draw a square in the middle of the map
    map.add_rectangle(40, 40, 20, 20)

    # reveal bottom half of the map
    map.ogm[0:50, :] = map.occupancy[0:50, :]

    agents = []
    agents.append(Agent(25, 20, 5, COMMS_RADIUS))
    agents.append(Agent(50, 20, 5, COMMS_RADIUS))
    agents.append(Agent(75, 20, 5, COMMS_RADIUS))

    sim = Simulator(map, agents, None)

    sim.update_plot()


    plt.show()

    # perform djikstras on the map for agent 0
    # agent_0_cost = djikstra(map, (25, 20))

    # show the cost map
    # fig, ax = plt.subplots()
    # ax.imshow(agent_0_cost, cmap='gray', origin='lower')
    # plt.show()


    # get the RHS axis
    # ax1 = plt.gcf().axes[1]

    opt_assigmnents = calc_optimal_assignments(sim)

    for i, j in opt_assigmnents:
        ax1.plot(i, j, 'ro')

    plt.show()






if __name__ == "__main__":
    basic_opt_example()