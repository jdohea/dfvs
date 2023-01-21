"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

source: https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1#file-monte_carlo_tree_search-py
"""
import random
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
import math
from dfvs_basic_branching import DFVS


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."
    killer = None
    current_i = None
    def __init__(self, exploration_weight=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = dict()  # children of each node
        self.exploration_weight = exploration_weight

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        # if node.is_terminal():
        #     raise RuntimeError(f"choose called on terminal node {node}")

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
            if node not in self.children or not self.children[node] or node.terminal_node:
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
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            if MCTS.killer.exit_now:
                if len(MCTS.killer.final_sets[MCTS.current_i]) > DFVS.m_dfvs_len:
                    MCTS.killer.final_sets[MCTS.current_i] = DFVS.m_dfvs
                MCTS.killer.give_answer()

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        # assert all(n in self.children for n in self.children[node])

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


class UpperBoundDFVSNode(Node):
    # global variables
    V_size_of_graph = None

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
        if MCTS.killer.exit_now:
            if len(MCTS.killer.final_sets[MCTS.current_i]) > DFVS.m_dfvs_len:
                MCTS.killer.final_sets[MCTS.current_i] = DFVS.m_dfvs
            MCTS.killer.give_answer()
        new_depth = self.depth + 1
        if new_depth >= DFVS.m_dfvs_len:
            return {}
        all_children = set()
        cur_dfvs = self.dfvs()
        candidate_child_elements = DFVS.graph_has_cycles(cur_dfvs)
        new_len_vertices_in_graph = UpperBoundDFVSNode.V_size_of_graph - new_depth
        for element in candidate_child_elements:
            cur_dfvs.add(element)
            new_is_terminal = UpperBoundDFVSNode.calc_is_terminal(cur_dfvs)
            if not new_is_terminal:
                reward = 0
            elif new_depth >= DFVS.m_dfvs_len:
                reward = 0
            else:
                reward = UpperBoundDFVSNode.calc_reward(new_len_vertices_in_graph)
            all_children.add(
                UpperBoundDFVSNode(element, new_is_terminal, reward, self, new_depth, hash(frozenset(cur_dfvs))))
            cur_dfvs.remove(element)
        return all_children

    @staticmethod
    def calc_is_terminal(dfvs, shuffle_list=False):
        "Returns True if the node has no children"
        ans = DFVS.graph_has_cycles(dfvs, shuffle_list=shuffle_list)
        if len(dfvs) >= DFVS.m_dfvs_len:
            return True
        elif ans:
            return False
        else:
            return True

    @staticmethod
    def calc_reward(num_vertices_in_graph):
        return num_vertices_in_graph

    def find_random_child(self):
        "All possible successors of this board state"
        if MCTS.killer.exit_now:
            if len(MCTS.killer.final_sets[MCTS.current_i]) > DFVS.m_dfvs_len:
                MCTS.killer.final_sets[MCTS.current_i] = DFVS.m_dfvs
            MCTS.killer.give_answer()
        new_depth = self.depth + 1
        # todo possible improvement but there will be less cycles found as quick?
        cur_dfvs = self.dfvs()
        new_len_vertices_in_graph = UpperBoundDFVSNode.V_size_of_graph - new_depth
        cycle = DFVS.graph_has_cycles(cur_dfvs)
        new_dfvs_element = random.sample(cycle, 1)[0]
        cur_dfvs.add(new_dfvs_element)
        new_is_terminal = UpperBoundDFVSNode.calc_is_terminal(cur_dfvs)
        if not new_is_terminal:
            reward = 0
        else:
            reward = UpperBoundDFVSNode.calc_reward(new_len_vertices_in_graph)
        return UpperBoundDFVSNode(new_dfvs_element, new_is_terminal, reward, self, new_depth, hash(frozenset(cur_dfvs)))

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


class LowerBoundCyclePackNode(Node):
    # below initiations are for the root
    parent_node = None
    depth = 0
    MAX_PACKING_NUMBER = 0

    def __init__(self, this_nodes_uniques_dfvs_element, is_terminal, parent_node, depth, hash_num):
        self.terminal_node = is_terminal
        self.parent_node = parent_node
        self.this_nodes_uniques_dfvs_element = this_nodes_uniques_dfvs_element
        self.depth = depth
        self.hash_num = hash_num

    def cycle_pack(self):
        if self.parent_node is None:
            return set()
        x = self.parent_node.cycle_pack()
        x.add(self.this_nodes_uniques_dfvs_element)
        return x

    def find_children(self):
        "All possible successors of this board state"
        children = set()
        new_depth = self.depth + 1
        cur_pack = self.cycle_pack()
        child_sets = LowerBoundCyclePackNode.all_disjoint_sets(cur_pack, MontyCarlaSolver.set_of_cycles)
        for child in child_sets:
            cur_pack.add(child)
            is_terminal = LowerBoundCyclePackNode.calc_is_terminal(cur_pack, new_depth)
            children.add(LowerBoundCyclePackNode(child, is_terminal, self, new_depth, hash(frozenset(cur_pack))))

        return children

    @staticmethod
    def all_disjoint_sets(cur_pack, all_cycles):
        disjoint_sets = set()
        for cyc in all_cycles:
            is_disjoint = True
            for x in cur_pack:
                is_disjoint = x.isdisjoint(cyc)
                if not is_disjoint:
                    break
            if is_disjoint:
                disjoint_sets.add(cyc)
        return disjoint_sets

    @staticmethod
    def calc_is_terminal(cur_pack, depth):
        "Returns True if the node has no children"
        dfvs = set()
        for x in cur_pack:
            dfvs.update(x)
        ans = DFVS.graph_has_cycles(dfvs)
        MontyCarlaSolver.set_of_cycles.add(frozenset(ans))
        if ans:
            return False
        else:
            if depth > LowerBoundCyclePackNode.MAX_PACKING_NUMBER:
                LowerBoundCyclePackNode.MAX_PACKING_NUMBER = depth
            return True

    def find_random_child(self):
        new_depth = self.depth + 1
        cur_pack = self.cycle_pack()
        child_sets = LowerBoundCyclePackNode.all_disjoint_sets(cur_pack, MontyCarlaSolver.set_of_cycles)
        child = random.sample(child_sets, 1)[0]
        cur_pack.add(child)
        is_terminal = LowerBoundCyclePackNode.calc_is_terminal(cur_pack, new_depth)
        return LowerBoundCyclePackNode(child, is_terminal, self, new_depth, hash(frozenset(cur_pack)))

    def is_terminal(self):
        "Returns True if the node has no children"
        return self.terminal_node

    def reward(self):
        "Assumes `self` is terminal node. blah1=win, 0=loss, .5=tie, etc"
        return self.depth

    def __hash__(self):
        "Nodes must be hashable"
        return self.hash_num

    def __eq__(node1, node2):
        "Nodes must be comparable"
        return node1.hash_num == node2.hash_num


class MontyCarlaSolver:
    name = 'MCTS'
    set_of_cycles = set()

    @staticmethod
    def exact_solver(path):
        graph = DFVS.load_graph_binary_matrix_from_row_to_col(path)
        DFVS.reset_DFVS_basic(graph)
        number_of_vertices = len(graph)
        UpperBoundDFVSNode.V_size_of_graph = number_of_vertices
        is_terminal = UpperBoundDFVSNode.calc_is_terminal(set())
        upper_node = UpperBoundDFVSNode(None, is_terminal,
                                        UpperBoundDFVSNode.calc_reward(num_vertices_in_graph=number_of_vertices), None,
                                        0, hash(frozenset(set())))
        mcts_upper = MCTS()
        # mcts_lower = MCTS()
        # is_terminal = LowerBoundCyclePackNode.calc_is_terminal(set(), 0)
        # lower_node = LowerBoundCyclePackNode(None, is_terminal, None, 0, hash(frozenset(set())))

        i = 0
        while i < 99:
            before = time.time()
            mcts_upper.do_rollout(upper_node)
            i += 1
            after = time.time()
        return DFVS.m_dfvs_len

    @staticmethod
    def solve_graph(graph, time_allocated, current_i, do_one_rollout = True):
        MCTS.current_i = current_i
        DFVS.reset_DFVS_g(graph)
        number_of_vertices = len(graph.graph)
        UpperBoundDFVSNode.V_size_of_graph = number_of_vertices
        is_terminal = UpperBoundDFVSNode.calc_is_terminal(set())
        upper_node = UpperBoundDFVSNode(None, is_terminal,
                                        UpperBoundDFVSNode.calc_reward(num_vertices_in_graph=number_of_vertices), None,
                                        0, hash(frozenset(set())))
        mcts_upper = MCTS()
        # mcts_lower = MCTS()
        # is_terminal = LowerBoundCyclePackNode.calc_is_terminal(set(), 0)
        # lower_node = LowerBoundCyclePackNode(None, is_terminal, None, 0, hash(frozenset(set())))

        before = time.time()
        if do_one_rollout:
            mcts_upper.do_rollout(upper_node)
            return DFVS.m_dfvs

        after = 0
        while after - before < time_allocated:
            mcts_upper.do_rollout(upper_node)
            if MCTS.killer.exit_now:
                if len(MCTS.killer.final_sets[MCTS.current_i]) > DFVS.m_dfvs_len:
                    MCTS.killer.final_sets[MCTS.current_i] = DFVS.m_dfvs
                MCTS.killer.give_answer()
            MCTS.killer.update_final_sets(current_i, DFVS.m_dfvs)
            after = time.time()
        return DFVS.m_dfvs


if __name__ == "__main__":
    # sys.setrecursionlimit(10 ** 6)
    print(MontyCarlaSolver.exact_solver('graphs/exact_public/e_003'))
