import gurobipy as gp
from gurobipy import GRB
import numpy as np

obj = [0, 1, 2, 3, 4, 5]
cycles = [{0, 1, 2}, {3, 4}]

import scipy.sparse as sp

try:
    #  maximize
    #        x +   y + 2 z
    #  subject to
    #        x + 2 y + 3 z <= 4
    #        x +   y       >= 1
    #        -x -y <= -1
    #        x, y, z binary
    # Create a new model
    obj = [0, 1, 2, 3, 4, 5]
    cycles = [{0, 1, 2}, {3, 4}]

    m = gp.Model("linear-model")

    # Create variables
    x = m.addMVar(shape=len(obj), vtype=GRB.BINARY)

    # Set objective
    # same again for strong components len(obj) is the total vertices in og graph, let blah below equal 1 where the
    blah = np.array([1] * len(obj))
    m.setObjective(blah @ x, GRB.MINIMIZE)
    M_consts = []
    for cycle in cycles:
        # remember if using strong components, let len(obj) be the number of vertices in the original graph
        tmp = np.array([0] * len(obj))
        for v in cycle:
            tmp[v] = -1
        m.addConstr(tmp @ x <= -1)
    m.optimize()
    print(x.X)

    print('Obj: %g' % m.ObjVal)

except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
