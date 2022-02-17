import pygmo as pg
prob = pg.problem(pg.rosenbrock(dim = 5))
print(prob)