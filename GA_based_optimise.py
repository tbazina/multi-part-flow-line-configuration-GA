import pygmo as pg
from pygmo import sga

# Maksimum generations of a run
MAXGEN = 1000
# Mutation probability
pm = 0.5
# Crossover probability
cm = 0.3

# User defined algorithm (UDA)
algo = pg.algorithm(
    sga(
        gen=MAXGEN,
        cr=cm,
        m=pm,
        selection='truncated'
    )
)