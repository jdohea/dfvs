import math
import time
import Graph

class DFVS:
    name = 'BasicBranching'
    number_of_vertices = 0
    m_dfvs = set()
    m_dfvs_len = math.inf
    lower_bound = 0
    g = None

    def __init__(self, graph_obj):
        self.lower_bound = 0
        self.number_of_vertices = len(graph_obj.graph)
        self.g = graph_obj
        self.m_dfvs = set(self.g.keys())
        self.m_dfvs_len = len(self.m_dfvs)

    @staticmethod
    def reset_DFVS_basic(graph_binary_array):
        DFVS.lower_bound = 0
        DFVS.number_of_vertices = len(graph_binary_array)
        DFVS.g = Graph.Graph(len(graph_binary_array))
        DFVS.m_dfvs = set(range(DFVS.number_of_vertices))
        DFVS.m_dfvs_len = math.inf
        for p in range(len(graph_binary_array)):
            for q in range(len(graph_binary_array[0])):
                if graph_binary_array[p][q]:
                    DFVS.g.addEdge(p, q)


    @staticmethod
    def reset_DFVS_g(graph_obj):
        DFVS.lower_bound = 0
        DFVS.number_of_vertices = len(graph_obj.graph)
        DFVS.m_dfvs_len = math.inf
        DFVS.g = graph_obj
        DFVS.m_dfvs = set(range(DFVS.number_of_vertices))

    @staticmethod
    def update_m_dfvs(dfvs):
        if len(dfvs) < DFVS.m_dfvs_len:
            DFVS.m_dfvs = dfvs.copy()
            DFVS.m_dfvs_len = len(dfvs)
            return True
        return False

    @staticmethod
    def graph_has_cycles(dfvs, shuffle_list=False):
        is_cyclic = DFVS.g.is_cyclic(dfvs=dfvs, shuffle_list=shuffle_list)
        if not is_cyclic:
            DFVS.update_m_dfvs(dfvs)
        return is_cyclic

    @staticmethod
    def brute_force_search_for_min_dfvs(backtrack_index, dfvs=set(),
                                        depth=0):
        # upper bound on branching
        # if we are going for experiments change this to ">="
        # todo put m_dfvs as a global variable so it works for all
        # todo change to the c++ code version so it looks at smaller sets first and quits on the first set it finds.
        if depth >= len(DFVS.m_dfvs):
            return DFVS.m_dfvs
        if not DFVS.graph_has_cycles(dfvs=dfvs):
            return DFVS.m_dfvs
        new_depth = depth + 1
        for p in range(backtrack_index, DFVS.number_of_vertices):
            dfvs.add(p)
            DFVS.brute_force_search_for_min_dfvs(dfvs=dfvs, depth=new_depth,backtrack_index=p+1)
            dfvs.remove(p)

        return DFVS.m_dfvs

    @staticmethod
    # todo ignore comment lines https://pacechallenge.org/2022/tracks/#input-format
    #   question for when implementing code for optil, will i get path to file or file as a string?
    def load_graph_binary_matrix_from_row_to_col(path_to_graph):

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
        # todo use graph type that is in CycleChecker, this is just a long way around.
        with open(path_to_graph) as file_in:
            file = file_in.readlines()
        line1 = file[0].split(' ')
        number_of_vertices = int(line1[0])
        g = [ [0]*number_of_vertices for _ in range(number_of_vertices) ]
        for i in range(number_of_vertices):
            line = file[i + 1].strip()
            line = line.split(' ')
            if not line[0] == '':
                line = list(map(int, line))
                for j in line:
                    g[i][j - 1] = 1
        return g


    # count how many cycles there are then before and after we take a node out.

    # Each exact solver method returns blah1 mdfvs
    # todo take into account special case of empty dfvs, can't have a full dfvs
    #    heuristic perhaps, just start the m_dfvs as the whole set
    @staticmethod
    def exact_solver(path):
        graph = DFVS.load_graph_binary_matrix_from_row_to_col(path)
        DFVS.reset_DFVS_basic(graph)
        return DFVS.brute_force_search_for_min_dfvs(backtrack_index=0)


if __name__ == "__main__":
    before = time.time()
    print(DFVS.exact_solver('graphs/exact_public_2/e_001'))
    after = time.time()
    print(before - after)
