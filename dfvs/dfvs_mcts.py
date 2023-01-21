"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

source: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1#file-monte_carlo_tree_search-py
"""
import random
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from dfvs_basic_branching import DFVS


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                # node is either unexplored or terminal
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if node.is_terminal():
                reward = node.reward()
                ## todo this should be jst erward
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            "Upper confidence bound for trees"
            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. blah1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True


class DFVSNode(Node):
    # below initiations are for the root
    parent_node = None
    depth = 0
    V_size_of_graph = None
    graph_vertices = None

    def __init__(self, this_nodes_uniques_dfvs_element, is_terminal, node_reward, parent_node, depth, hash_num):
        self.terminal_node = is_terminal
        self.node_reward = node_reward
        self.parent_node = parent_node
        self.this_nodes_uniques_dfvs_element = this_nodes_uniques_dfvs_element
        self.depth = depth
        self.hash_num = hash_num

    def dfvs(self):
        if self.parent_node is None:
            return set()
        x = self.parent_node.dfvs()
        x.add(self.this_nodes_uniques_dfvs_element)
        return x

    def find_children(self):
        "All possible successors of this board state"
        new_depth = self.depth + 1
        cur_dfvs = self.dfvs()
        vertices_in_graph = DFVSNode.graph_vertices.difference(cur_dfvs)
        new_children = set()
        new_len_vertices_in_graph = len(vertices_in_graph) - 1
        for x in vertices_in_graph:
            tmp_dfvs = cur_dfvs.copy()
            tmp_dfvs.add(x)
            new_is_terminal = DFVSNode.calc_is_terminal(tmp_dfvs)
            new_children.add(
                DFVSNode(x, new_is_terminal, DFVSNode.calc_reward(new_len_vertices_in_graph, new_is_terminal), self,
                         new_depth, hash(frozenset(tmp_dfvs))))

        return new_children

    @staticmethod
    def calc_is_terminal(dfvs):
        "Returns True if the node has no children"
        ans = DFVS.graph_has_cycles(dfvs)
        if ans:
            return False
        else:
            return True

    @staticmethod
    def calc_reward(num_vertices_in_graph, is_terminal):
        return num_vertices_in_graph * 100 * is_terminal

    def find_random_child(self):
        "All possible successors of this board state"
        new_depth = self.depth + 1
        cur_dfvs = self.dfvs()
        vertices_in_graph = DFVSNode.graph_vertices.difference(cur_dfvs)
        new_children = set()
        new_len_vertices_in_graph = len(vertices_in_graph) - 1
        x = random.sample(vertices_in_graph, 1)[0]
        tmp_dfvs = cur_dfvs.copy()
        tmp_dfvs.add(x)
        new_is_terminal = DFVSNode.calc_is_terminal(tmp_dfvs)
        new_child = DFVSNode(x, new_is_terminal, DFVSNode.calc_reward(new_len_vertices_in_graph, new_is_terminal), self,
                             new_depth, hash(frozenset(tmp_dfvs)))

        return new_child

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.terminal_node

    def reward(self):
        "Assumes `self` is terminal node. blah1=win, 0=loss, .5=tie, etc"
        return self.node_reward

    def __hash__(self):
        "Nodes must be hashable"
        return self.hash_num

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return node1.hash_num == node2.hash_num


class MCTSSolver:
    name = 'MCTS'

    @staticmethod
    def exact_solver(path):
        graph = DFVS.load_graph_binary_matrix_from_row_to_col(path)
        DFVS.reset_DFVS_basic(graph)
        number_of_vertices = len(graph)
        DFVSNode.V_size_of_graph = number_of_vertices
        DFVSNode.graph_vertices = set(range(number_of_vertices))
        DFVSNode.m_dfvs = [[0] * number_of_vertices for _ in range(number_of_vertices)]
        DFVSNode.m_dfvs_length = number_of_vertices
        is_terminal = DFVSNode.calc_is_terminal(set())
        node = DFVSNode(None, is_terminal,
                        DFVSNode.calc_reward(num_vertices_in_graph=number_of_vertices, is_terminal=is_terminal), None,
                        0, hash(frozenset(set())))
        mcts = MCTS()
        i = 0
        while i < 50:
            mcts.do_rollout(node)
            i += 1
        return len(DFVS.m_dfvs)


# todo a variation of mcts where I update the bounds as i search so i start marking nodes that have dfvs size
#   bigger than the current best as having a score of 0
if __name__ == "__main__":
    sys.setrecursionlimit(10 ** 6)
    print(MCTSSolver.exact_solver('/Users/home/PycharmProjects/dfvs/graphs/heuristic_public/h_195'))
