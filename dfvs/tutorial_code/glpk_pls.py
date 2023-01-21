from pulp import GLPK
from pulp import LpProblem, lpSum, LpVariable, LpBinary, LpMinimize, LpContinuous
# Create the model
blah = LpProblem(name='dfvs', sense=LpMinimize)
obj = [0, 1, 2, 3, 4, 5]
decisionVars = LpVariable.dicts('node', obj ,0,1, LpBinary)
cycles = [{0, 1, 2}, {3, 4}]
for cycle in cycles:
    blah += (lpSum([decisionVars[i] for i in cycle]) >= 1)
blah += lpSum(decisionVars)
# Solve the problem
status = blah.solve(solver=GLPK(msg=False))

# print(f"status: {blah.status}, {LpStatus[blah.status]}")
print(type(blah.objective.value()))
#
# for var in blah.variables():
#     print(f"{var.name}: {var.value()}")
#
# for name, constraint in blah.constraints.items():
#     print(f"{name}: {constraint.value()}")