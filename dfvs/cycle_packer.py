import math

from dfvs_basic_branching import DFVS
from cycle_checker import Graph


class CyclePacker:

    NUMBER_OF_TIMES_TO_CYCLE_PACK = 10000

    def initialize_and_find_lowerbound(self, path):
        graph = DFVS.load_graph_binary_matrix_from_row_to_col(path)
        DFVS.reset_DFVS_basic(graph)
        return self.find_lower_bound()

    def find_lower_bound(self):
        max_lower_bound = 0
        m_dfvs = None
        for i in range(CyclePacker.NUMBER_OF_TIMES_TO_CYCLE_PACK):
            lower_bound = 0
            dfvs = set()
            cycle = DFVS.g.is_cyclic(dfvs, shuffle_list=True)
            while cycle:
                lower_bound += 1
                dfvs.update(cycle)
                cycle = DFVS.g.is_cyclic(dfvs)
            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
            if DFVS.update_m_dfvs(dfvs):
                m_dfvs = dfvs.copy()
        return max_lower_bound


if __name__ == "__main__":
    packer = CyclePacker()
    print(packer.initialize_and_find_lowerbound('graphs/exact_public/e_003'))
