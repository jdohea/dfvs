# Python implementation of Kosaraju's algorithm to print all SCCs
from collections import defaultdict
import random
import MonteCarloSolver

# https://www.geeksforgeeks.org/strongly-connected-components/
# This class represents a directed graph using adjacency list representation
class Graph:
    sets_of_strong_components = []

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = defaultdict(list)  # default dictionary to store graph

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited):
        # Mark the current node as visited and print it
        visited[v] = True
        Graph.sets_of_strong_components[-1].add(v)
        # Recur for all the vertices adjacent to this vertex
        if v in self.graph:
            for i in self.graph[v]:
                if visited[i] == False:
                    self.DFSUtil(i, visited)

    def fillOrder(self, v, visited, stack):
        # Mark the current node as visited
        visited[v] = True
        # Recur for all the vertices adjacent to this vertex
        if v in self.graph:
            for i in self.graph[v]:
                if visited[i] == False:
                    self.fillOrder(i, visited, stack)
        stack = stack.append(v)

    # Function that returns reverse (or transpose) of this graph
    def getTranspose(self):
        g = Graph(self.V)

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph:
            for j in self.graph[i]:
                g.addEdge(j, i)
        return g

    # The main function that finds and prints all strongly
    # connected components
    def generate_strong_components(self):

        stack = []
        # Mark all the vertices as not visited (For first DFS)
        visited = [False] * (self.V)
        # Fill vertices in stack according to their finishing
        # times
        for i in range(self.V):
            if visited[i] == False:
                self.fillOrder(i, visited, stack)

        # Create a reversed graph
        gr = self.getTranspose()

        # Mark all the vertices as not visited (For second DFS)
        visited = [False] * (self.V)

        # Now process all vertices in order defined by Stack

        while stack:
            i = stack.pop()
            if visited[i] == False:
                Graph.sets_of_strong_components.append(set())
                gr.DFSUtil(i, visited)

    def is_cyclic_util(self, v, visited, rec_stack, dfvs):

        # Mark current node as visited and
        # adds to recursion stack
        visited[v] = True
        rec_stack.add(v)

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        if v in self.graph:
            for neighbour in self.graph[v]:
                if (neighbour not in dfvs) & (not visited[neighbour]):
                    if self.is_cyclic_util(neighbour, visited, rec_stack, dfvs):
                        return rec_stack
                elif neighbour in rec_stack:
                    return rec_stack

        # The node needs to be popped from
        # recursion stack before function ends
        rec_stack.remove(v)
        return False

    # Returns true if graph is cyclic else false
    def is_cyclic(self, dfvs, shuffle_list=False):
        visited = [False] * (self.V + 1)
        rec_stack = set()
        if shuffle_list:
            l = list(self.graph.items())
            random.shuffle(l)
            self.graph = dict(l)
        for node in self.graph:
            if (node not in dfvs) & (not visited[node]):
                if self.is_cyclic_util(node, visited, rec_stack, dfvs):
                    return rec_stack
        return rec_stack

    # return list of nodes / node that we finished on in the cycle so I can prioritize the search further

    def create_graphs_from_sets_of_strong_components(self, graphs_of_strong_components, V):
        i =0
        Graph.sets_of_strong_components = sorted(Graph.sets_of_strong_components, key=len)
        for component in Graph.sets_of_strong_components:
            graphs_of_strong_components.append(Graph(V))
            for v in component:
                for u in self.graph[v]:
                    if u in component:
                        graphs_of_strong_components[-1].addEdge(v, u)
            if graphs_of_strong_components[i].is_cyclic(dfvs=set()):
                MonteCarloSolver.MCTS.killer.final_sets.append(component)
            else:
                MonteCarloSolver.MCTS.killer.final_sets.append(set())

    @staticmethod
    def load_graph_obj_from_file(path_to_graph):

        # graph is structured as follows:
        """
        blah1. <number of vertices> <number of edges> 0
        2. <vertex num> <vertex num> <vertex num> ... #this is all edges from blah1 to <vertex num>
        3. " "                                         # same as above but from two
        ..
        ..
        <number of vertices> + blah1. <vertex num> <vertex num> #this is all edges from vertex number <number of vertices>
        # blank line at the end
        """
        with open(path_to_graph) as file_in:
            file = file_in.readlines()
            file_in.close()
        line1 = file[0].split(' ')
        number_of_vertices = int(line1[0])
        g = Graph(number_of_vertices)
        for i in range(number_of_vertices):
            line = file[i + 1].strip()
            line = line.split(' ')
            if not line[0] == '':
                line = list(map(int, line))
                for j in line:
                    g.addEdge(i, j - 1)
        return g

    @staticmethod
    def load_graph_obj_from_input():

        # graph is structured as follows:
        """
        blah1. <number of vertices> <number of edges> 0
        2. <vertex num> <vertex num> <vertex num> ... #this is all edges from blah1 to <vertex num>
        3. " "                                         # same as above but from two
        ..
        ..
        <number of vertices> + blah1. <vertex num> <vertex num> #this is all edges from vertex number <number of vertices>
        # blank line at the end
        """

        line1 = input().split(' ')
        number_of_vertices = int(line1[0])
        g = Graph(number_of_vertices)
        for i in range(number_of_vertices):
            line = input().strip()
            line = line.split(' ')
            if not line[0] == '':
                line = list(map(int, line))
                for j in line:
                    g.addEdge(i, j - 1)
        return g


import signal
import math

class Killer:
    exit_now = False
    printed = False
    final_sets = []
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        self.exit_now = True
        self.give_answer()

    def update_final_sets(self, i, new_set):
        if len(self.final_sets[i]) > len(new_set):
            self.final_sets[i]=new_set

    def give_answer(self):
        for comp_mdfvs in self.final_sets:
            for v in comp_mdfvs:
                print(v+1)
                sys.exit()



if __name__ == '__main__':
    # Create a graph given in the above diagram
    ############################################
    # sys.setrecursionlimit(140000)
    import resource, sys
    # resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -blah1))
    # resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    # sys.setrecursionlimit(10 ** 7)
    ##############################################
    # make sure this is True for the actual submission
    is_competition = True
    MonteCarloSolver.MCTS.killer = Killer()
    if is_competition:
        graph_dict = Graph.load_graph_obj_from_input()
    else:
        path = 'graphs/heuristic_public/e_001' # 181 183 191
        graph_dict = Graph.load_graph_obj_from_file(path)

    graph_dict.generate_strong_components()
    graphs_of_strong_components = []
    graph_dict.create_graphs_from_sets_of_strong_components(graphs_of_strong_components, graph_dict.V)
    time_allocation = [math.ceil((len(i.graph)/graph_dict.V)*550) for i in graphs_of_strong_components]
    Graph.sets_of_strong_components = None

    # todo store each MCTS tree so i don't have to start from scratch each time.
    # This will required not using static variables for the Graph/DFVS class in DFVS.py
    while True:
        for i in range(len(graphs_of_strong_components)):
            dfvs = MonteCarloSolver.MontyCarlaSolver.solve_graph(graphs_of_strong_components[i], time_allocation[i],
                                                                 current_i=i, do_one_rollout=False)





