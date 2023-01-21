import math
import sys
import time
from collections import defaultdict
import random

from discord_webhook import DiscordWebhook

import MonteCarloSolver
from pulp import GLPK, GUROBI, CPLEX_PY
import traceback
from pulp import LpProblem, lpSum, LpVariable, LpBinary, LpMinimize, LpContinuous
import pandas as pd
import os


# https://www.geeksforgeeks.org/strongly-connected-components/
# This class represents a directed graph using adjacency list representation
# gc.set_threshold(1,1,1)
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

        # Recur for all neighbours
        # if any neighbour is visited and in
        # recStack then graph is cyclic
        if v in self.graph:
            rec_stack.add(v)
            for neighbour in self.graph[v]:
                if (neighbour not in dfvs) & (not visited[neighbour]):
                    if self.is_cyclic_util(neighbour, visited, rec_stack, dfvs):
                        return rec_stack
                elif neighbour in rec_stack:
                    return rec_stack
            rec_stack.remove(v)
        # The node needs to be popped from
        # recursion stack before function ends

        return False

    # Returns true if graph is cyclic else false
    def is_cyclic(self, dfvs):
        visited = [False] * (self.V + 1)
        rec_stack = set()
        for node in self.graph:
            if (node not in dfvs) & (not visited[node]):
                if self.is_cyclic_util(node, visited, rec_stack, dfvs):
                    del visited, dfvs
                    return rec_stack

        return rec_stack

    # return list of nodes / node that we finished on in the cycle so I can prioritize the search further

    def generate_and_preprocess_graph_comps(self, graphs_of_strong_components, dfvs_s_at_root_nodes_per_comp, V):
        for component in Graph.sets_of_strong_components:
            if len(component) == 1:
                pass
            else:
                graphs_of_strong_components.append(Graph(V))
                dfvs_s_at_root_nodes_per_comp.append(set())
                reversed_g = Graph(V)
                for v in component:
                    for u in self.graph[v]:
                        if u in component:
                            graphs_of_strong_components[-1].addEdge(v, u)
                            reversed_g.addEdge(u, v)

                keys = list(reversed_g.graph.keys())
                for i in keys:
                    if len(reversed_g.graph[i]) == 1:
                        if reversed_g.graph[i][0] == i:
                            dfvs_s_at_root_nodes_per_comp[-1].add(i)
                        else:
                            Graph.remove_j(graphs_of_strong_components[-1].graph, reversed_g.graph, i)

                keys = list(graphs_of_strong_components[-1].graph)
                for i in keys:
                    if len(graphs_of_strong_components[-1].graph[i]) == 1:
                        if graphs_of_strong_components[-1].graph[i][0]==i:
                            dfvs_s_at_root_nodes_per_comp[-1].add(i)
                        else:
                            Graph.remove_j(reversed_g.graph, graphs_of_strong_components[-1].graph, i)


    @staticmethod
    def remove_j(g_1in, g_1out, j_rm):
        # put the 1 ins back to the u on the (u,v) single in edge
        edges_to_move_back = g_1in[j_rm]
        node_to_move_to = g_1out[j_rm][0]
        for e in edges_to_move_back:
            if e not in g_1in[node_to_move_to]:
                g_1in[node_to_move_to].append(e)
        g_1in[node_to_move_to].remove(j_rm)
        # put the 1 outs in edges forward a node
        for e in edges_to_move_back:
            if node_to_move_to in g_1out[e]:
                g_1out[e].remove(j_rm)
            else:
                ind = g_1out[e].index(j_rm)
                g_1out[e][ind] = node_to_move_to
        g_1out.pop(j_rm)
        g_1in.pop(j_rm)


        # if graphs_of_strong_components[i].is_cyclic(dfvs=set()):
        #     MontyCarlaSolver.MCTS.killer.final_sets.append(component)
        # else:
        #     MontyCarlaSolver.MCTS.killer.final_sets.append(set())

    @staticmethod
    def load_graph_obj_from_file(path_to_graph):
        # graph is structured as follows:
        """
        1. <number of vertices> <number of edges> 0
        2. <vertex num> <vertex num> <vertex num> ... #this is all edges from 1 (2-1) to <vertex num>
        3. " "                                         # same as above but from two
        ..
        ..
        <number of vertices> + 1. <vertex num> <vertex num> #this is all edges from vertex <number of vertices> to <vertex num>
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





class Node:
    V_NUMBER_GLOBAL = None
    GLOBAL_UPPER_BOUND = 999999999
    GRAPH = None
    PRUNED_DFVS_S = {}
    MCTS_EXPLORATION_WEIGHT = math.sqrt(2)
    M_DFVS = set()
    SET_OF_CYCLES = set()
    DEEPEST_LEVEL_EXHAUSTED = 0
    PRUNED_DFVS_S = set()

    @staticmethod
    def reset(graph):
        Node.V_NUMBER_GLOBAL=len(graph.graph)
        Node.GRAPH = graph
        Node.M_DFVS = set()
        Node.GLOBAL_UPPER_BOUND =  len(graph.graph)
        Node.SET_OF_CYCLES = set()
        Node.PRUNED_DFVS_S = set()

    def __init__(self, parent_node, dfvs):
        self.parent_node = parent_node
        self.dfvs = dfvs

        # experiment: it could be worth continuing to do these as we cycle checks as it would add to the sets of cycles
        # if len(self.dfvs) >= Node.GLOBAL_UPPER_BOUND:
        #     self.candidate_children = {}
        #     self.exhausted = True
        # else:
        self.candidate_children = Node.GRAPH.is_cyclic(self.dfvs)
        if not self.candidate_children:
            # if it is exhausted this means it has been checked for a the best solution
            # never exhaust a node that has a new best solution because it needs to be rewarded in MCTS
            # experiment with limiting the search depth in MCTS to the current upper or if I should let in go deep
            # this would allow it to learn beyond the upper bound and seek out an area with more better results
            # as opposed to pruning off an area early
            Node.update_mdfvs(dfvs)
        else:
            Node.SET_OF_CYCLES.add(frozenset(self.candidate_children.copy()))
        self.children_initialized = []
        self.total_visits = 0
        self.sum_of_rewards = 0

    def initialize_random_child(self, bandb=False):
        # check before I call here for if there are candidate
        element = self.candidate_children.pop()
        tmp = set(self.dfvs)
        tmp.add(element)
        tmp = frozenset(tmp)
        # if tmp in Node.NODES:
        #     self.children_initialized.append(Node.NODES[tmp])
        # else:
        node = Node(self, tmp)
        if len(self.dfvs) < 25 or bandb:
            self.children_initialized.append(node)
        return node

    def has_candidate_children(self):
        return len(self.candidate_children)

    def mcts_expand_node(self):
        reward = 0
        # don't expand beyond upper bound
        # if len(self.dfvs) >= Node.GLOBAL_UPPER_BOUND:
        #     Node.NODES[self.dfvs].total_visits += 1
        #     if not self.exhausted:
        #         reward = len(Node.GRAPH.graph.keys()) - len(self.dfvs)
        #
        #     Node.NODES[self.dfvs].sum_of_rewards += reward
        #     return reward
        if not self.has_candidate_children() and not self.children_initialized:
            self.total_visits += 1
            reward = 1.0 - (len(self.dfvs) / Node.V_NUMBER_GLOBAL)
            self.sum_of_rewards += reward
            return reward
        # if there are candidate children this dfvs was invalid -> cyclic
        elif self.has_candidate_children():
            reward = self.initialize_random_child().mcts_expand_node()
        else:
            # if len(self.dfvs) > Node.DEEPEST_LEVEL_EXHAUSTED:
            #     Node.DEEPEST_LEVEL_EXHAUSTED = len(self.dfvs)
            reward = self.uct_select().mcts_expand_node()

        self.sum_of_rewards += reward
        self.total_visits += 1
        return reward

    @staticmethod
    def is_subset_of_pruned_node(dfvs):
        for p in Node.PRUNED_DFVS_S:
            if p.issubset(dfvs):
                return True
        return False

    def branch(self):
        if len(self.dfvs) >= Node.GLOBAL_UPPER_BOUND:
            return

        while self.has_candidate_children():
            self.initialize_random_child(bandb=True)
        # number_of_lower_bounds_searches_allowed &
        # if ((
        #         len(self.dfvs) / Node.GLOBAL_UPPER_BOUND) > DO_LOWER_BOUND_RATIO_CONDITION) & (random.random() < PROBABILITY_OF_LOWER_BOUND_SEARCH):

        local_lowerbound = self.bound()
        lowerbound = local_lowerbound + len(self.dfvs)
        if lowerbound >= Node.GLOBAL_UPPER_BOUND:
            return
        # elif lowerbound-Node.GLOBAL_UPPER_BOUND > GAP_TO_UPPER_BOUND_RATIO:
        #     i = 0
        #     while i < NUMBER_OF_INTERNAL_MCTS_RUNS:
        #         self.mcts_expand_node()
        #         i += 1

        for child in self.children_initialized:
            child.branch()

    def bound(self):
        """
        choose one of the following lowerbounding methods:
        cycle_packing_lower_bound
        ilp
        lp_lower_bound
        :return: bound
        """
        return self.lp_lower_bound(ILP_OR_LP)  # optionn for LpBinary

    def greedy_cycle_packing_lower_bound(self):
        """
        should i use an MCTS cycle packing or a greedy one?
        start packing smaller cycles first? yes add them to a dfvs and then keep going - greedy
        another cycle packing that uses MCTS
        another cycle packing that uses random
        :return:
        """
        tmp = set(self.dfvs)
        i = 0
        l = list(Node.SET_OF_CYCLES)
        l.sort(key=len)
        for s in l:
            if tmp.isdisjoint(s):
                tmp.update(s)
                i += 1

        has_cycle = True
        while has_cycle:
            has_cycle = Node.GRAPH.is_cyclic(tmp)
            if has_cycle:
                Node.SET_OF_CYCLES.add(frozenset(has_cycle))
                tmp.update(has_cycle)
                i += 1

        return i

    def lp_lower_bound(self, lp_or_ilp):
        """
        If i create root lp see if i can add and remove constraints easily by checking if the consraint contains a given decision variable
        how do i add constraints from code-not a file?
        :return: math.ceil(lower bound from linear program relaxation)
        """
        return math.ceil(
            self.generic_solve_model(LpVariable.dicts('node', list(range(Node.V_NUMBER_GLOBAL)), 0, 1, lp_or_ilp)))

    def generic_solve_model(self, decision_vars):
        model = LpProblem(name='dfvs', sense=LpMinimize)
        for cycle in Node.SET_OF_CYCLES:
            if self.dfvs.isdisjoint(cycle):
                # remember if using strong components, let len(obj) be the number of vertices in the original graph
                model += (lpSum([decision_vars[i] for i in cycle]) >= 1)
        model += lpSum(decision_vars)
        model.solve(solver=GLPK(msg=False))  # GLPK(msg=False))
        return model.objective.value()

    def get_cycles_disjoint_with_dfvs(self):
        cycles = []
        for cycle in Node.SET_OF_CYCLES:
            if self.dfvs.isdisjoint(cycle):
                cycles.append(cycle)
        return cycles

    def uct_select(self):
        def uct(node):
            """Upper confidence bound for trees"""
            number_of_visits = node.total_visits
            # todo check what happens when the number of child visits is greater than the parents
            return node.sum_of_rewards / number_of_visits + (Node.MCTS_EXPLORATION_WEIGHT * math.sqrt(
                math.log(node.parent_node.total_visits) / number_of_visits))

        """Select a child of node, balancing exploration & exploitation"""
        # All children of node should already be expanded:
        return max(self.children_initialized, key=uct)

    @staticmethod
    def write_cycles_to_file(name = 'blah'):
        name = name+'_cycles'
        file = open(name, 'w+')
        for cycle in Node.SET_OF_CYCLES:
            line = ''
            for node in cycle:
                line += str(node + 1) + ' '
            file.write(line + '\n')
        file.close()

    @staticmethod
    def update_mdfvs(dfvs):
        if len(dfvs)< Node.GLOBAL_UPPER_BOUND:
            Node.GLOBAL_UPPER_BOUND = len(dfvs)
            Node.M_DFVS = dfvs


def solve_with_strong_components(path):
    # make sure this is True for the actual submission

    graph_dict = Graph.load_graph_obj_from_file(path)
    graph_dict.generate_strong_components()
    Node.V_NUMBER_GLOBAL = graph_dict.V
    graphs_of_strong_components = []
    root_node_dfvs_s = []
    graph_dict.generate_and_preprocess_graph_comps(graphs_of_strong_components, root_node_dfvs_s, graph_dict.V)
    Graph.sets_of_strong_components = []

    final_sets = []
    m_dfvs_size = 0
    mcts_time = []

    def blah(g):
        return len(g.graph)

    graphs_of_strong_components = sorted(graphs_of_strong_components, key=blah)

    for g in graphs_of_strong_components:
        mcts_time.append(MCTS_TIME*(len(g.graph)/graph_dict.V))
    for j in range(len(graphs_of_strong_components)):
        graph = graphs_of_strong_components[j]
        Node.reset(graph)
        root_node = Node(None, frozenset(root_node_dfvs_s[j]))
        before = time.time()
        after = 0
        while after - before < mcts_time[j]:
            root_node.mcts_expand_node()
            after = time.time()
        root_node.branch()
        final_sets.append(Node.M_DFVS)
        m_dfvs_size += Node.GLOBAL_UPPER_BOUND

    # give_answer(final_sets)
    return m_dfvs_size


def give_answer(final_sets):
    for comp_mdfvs in final_sets:
        for v in comp_mdfvs:
            print(v + 1)
            sys.exit()


# MCTS_TIME = float(os.getenv('MCTS_TIME'))
# DO_LOWER_BOUND_RATIO_CONDITION = float(os.getenv('DO_LOWER_BOUND_RATIO_CONDITION'))
# ILP_OR_LP = os.getenv('ILP_OR_LP')

MCTS_TIME = 0
DO_LOWER_BOUND_RATIO_CONDITION = 0
ILP_OR_LP = LpBinary

DONT_DO_LOWER_BOUND_RATIO_CONDTION = 1
PROBABILITY_OF_LOWER_BOUND_SEARCH = 0.05
GAP_TO_UPPER_BOUND_RATIO = 0.9
NUMBER_OF_INTERNAL_MCTS_RUNS = 0


def send_discord_message(message):
    url = os.getenv['discord_webhood_url']
    webhook = DiscordWebhook(url=url, rate_limit_retry=True, content=message)
    return webhook.execute()


def store_sccs():
    direc = os.getcwd() + '/graphs/exact_public/'
    dirs = os.listdir(direc)
    dirs = sorted(dirs)
    id = 0
    with open(os.getcwd()+'/graphs/sccs/meta.csv', "w+") as grr:
        grr.write('graph_name, vertices, edges\n')
    grr.close()
    for filename in dirs:
        path = direc + filename
        graph_dict = Graph.load_graph_obj_from_file(path)
        graph_dict.generate_strong_components()
        Node.V_NUMBER_GLOBAL = graph_dict.V
        graphs_of_strong_components = []
        root_node_dfvs_s = []
        graph_dict.generate_and_preprocess_graph_comps(graphs_of_strong_components, root_node_dfvs_s, graph_dict.V)
        Graph.sets_of_strong_components = []
        mappings = []
        edges = []
        for g in graphs_of_strong_components:
            edge=0
            mapping = dict()
            j=0
            for i in g.graph:
                edge += len(g.graph[i])
                mapping[i] = j
                j+=1
            edges.append(edge)
            mappings.append(mapping)
            with open(os.getcwd()+'/graphs/sccs/meta.csv', "a+") as grr:
                grr.write(str(id) + '_' + filename + ',' + str(len(g.graph)) + ',' + str(edge) + '\n')
            grr.close()
            with open(os.getcwd()+'/graphs/sccs/'+str(id)+'_'+filename, "w+") as f:
                f.write(str(len(g.graph))+' '+str(edge)+'\n')
                for i in g.graph:
                    for j in g.graph[i]:
                        f.write(str(mapping[j]+1)+' ')
                    f.write('\n')
                f.write('\n')
            f.close()
            id+=1
        grr.close()


def run(path):
    sys.setrecursionlimit(10 ** 7)
    # GRAPH, SOLVED, MCTS_TIME, LB_START_RATIO, LB_FINISH_RATIO, LB_OFTEN, INTERNAL_MTCS, NODE_PRIORITY, ILP_LP, STRONG_COMPONENTS, NOTES
    lp = "ilp"
    before_global = time.time()
    try:

        Node.GLOBAL_UPPER_BOUND = solve_with_strong_components(path)
        after_global = time.time()
        RES = path + ',' + str(after_global - before_global) + ',' + str(
            MCTS_TIME) + ',0,1,1,0,No,' + lp + ',No,' + str(
            Node.GLOBAL_UPPER_BOUND) + ',strong components\n'
        with open("results.csv", "a+") as f:
            f.write(RES)
            f.close()
        send_discord_message(RES)

    except Exception as e:
        raise e


# todo:
#   decision on whether or not to use strong components ( number of vertices, ratio of edges to vertices..)
#   decision on whether or not to go straight to branching or to use MCTS
if __name__ == '__main__':
    sys.setrecursionlimit(10 ** 7)
    # if ILP_OR_LP == 'LP':
    #     ILP_OR_LP = LpContinuous
    #     lp = 'LP'
    # else:
    #     lp = 'ILP'
    #     ILP_OR_LP = LpBinary
    # GRAPH, SOLVED, MCTS_TIME, LB_START_RATIO, LB_FINISH_RATIO, LB_OFTEN, INTERNAL_MTCS, NODE_PRIORITY, ILP_LP, STRONG_COMPONENTS, NOTES
    # run(sys.argv[1])
    graphs = ['e_191', 'e_193']
    for g in graphs:
        path = '/Users/home/PycharmProjects/dfvs/graphs/exact_public/'+g
        graph = Graph.load_graph_obj_from_file(path)
        Node.reset(graph)
        root_node = Node(None, frozenset())
        before = time.time()
        after = 0
        while after - before < 600:
            root_node.mcts_expand_node()
            after = time.time()
        Node.write_cycles_to_file(g)