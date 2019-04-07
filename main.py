import numpy as np
import pandas as pd
import pygmo as pg
import copy
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
import networkx as nx
from problem import (MPFL_configuration, search_sequence_numpy, FOC_decoding,
FOA_decoding, np_unique_impl)
from population import generate_individual
from algorithm import GA_based_optimize, crossover, mutation

"""
-------------------------- Options -------------------------- 
"""
# Pandas display options
pd.options.display.max_columns = 50
"""
------------------------------------------------------------------------------
"""

"""
-------------------------- Misc Functions -------------------------- 
"""
def str_sort_num(ls):
    """
    Sort input nested list using digits at the end and return only unique
    values.

    :param ls: nested lists containing strings and numbers

    :return: sorted list
    """
    flat_list = [item for sublist in ls for item in sublist]
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: convert(key[2:])
    return sorted(set(flat_list), key = alphanum_key)


def plot_graphs(G_j, O_j):
    """

    :param G_j:
    Lista grafova OC-a

    :return:
    Grafički prikaz grafova
    """
    node_plt_options = {
        'node_size': 900,
        'node_color': '0.9',
        'node_shape': 'o',
        'linewidths': 1.,
        'edgecolors': 'black',
    }
    edges_plt_options = {
        'width': 1.2,
        'edge_color': 'black',
        'style': 'solid',
        'arrows': True,
        'arrowstyle': '->',
        'arrowsize': 20.,
    }
    labels_plt_options = {
        'font_size': 9,
        'font_color': 'black',
        'font_weight': 'bold',

    }
    pos = [
        {
            O_j[0][0]: (3, 5)
        },
        {
            O_j[1][0]: (3, 5)
        }
    ]
    pos[0].update(
        {
            O_j[0][i]:(x, y) for i, x, y in zip(
            [4, 1, 6, 3], [1, 2, 4, 5], [3]*4
        )
        }
    )
    pos[0].update(
        {
            O_j[0][i]:(x, y) for i, x, y in zip(
            [5, 2], [1, 5], [1]*2
        )
        }
    )
    pos[1].update(
        {
            O_j[1][i]:(x, y) for i, x, y in zip(
            [4, 1, 6, 3, 7], [1, 2, 3, 4, 5], [3]*5
        )
        }
    )
    pos[1].update(
        {
            O_j[1][i]:(x, y) for i, x, y in zip(
            [10, 5, 9, 2, 8], [1, 2, 3, 4, 5], [1]*5
        )
        }
    )
    G_j_len = len(G_j)
    fig = plt.figure(figsize=(12, 4), dpi=300)
    for i in range(G_j_len):
        ax = fig.add_subplot(1, G_j_len, i+1)
        nx.draw_networkx_nodes(G_j[i], pos=pos[i], **node_plt_options)
        nx.draw_networkx_edges(G_j[i], pos=pos[i], **edges_plt_options)
        nx.draw_networkx_labels(G_j[i], pos=pos[i], **labels_plt_options)
    plt.show()
    return


def convergence_plot(algos, figsize=(16, 50)):
    """
    Plot konvergencije svih otoka

    :param algos:
    list of algorithms

    :return:
    """
    # Plot style
    plt.style.use('ggplot')
    # Extracting algorithms and logs
    # algos = [isl.get_algorithm() for isl in islands]
    udas = [alg.extract(GA_based_optimize) for alg in algos]
    logs = [uda.get_log() for uda in udas]
    # Creating figure
    fig = plt.figure(figsize=figsize)
    # Number of rows and columns
    cols = 2
    rows = (len(algos)//2) if not (len(algos) % 2) else len(algos)//2 + 1
    for num in range(1, len(algos)+1):
        ax = fig.add_subplot(rows, cols, num)
        ax.plot([entry[0] for entry in logs[num-1]],
                [entry[1] for entry in logs[num-1]],
                'k-',
                linewidth=2.0,
                label='Minimal capital cost')
        ax.plot([entry[0] for entry in logs[num-1]],
                [entry[2] for entry in logs[num-1]],
                'r--',
                linewidth=2.0,
                label='Average capital cost')
        ax.set_title('cm: {}, pm: {}'.format(udas[num-1].cm, udas[num-1].pm))
        ax.set_xlabel('Generation number')
        ax.set_ylabel('Capital cost (thousand USD)')
        ax.legend()
    plt.show()
    return


def get_best_individual_cm_pm(pops, algos):
    """
    Fitness najbolje jedinke svake populacije s pripadajućim crossover
    probability (cm) i mutation probability (pm)

    :param pops:
    List of populations

    :param algos:
    List of algorithms

    :return:
    """
    pops_best_f = [p.champion_f[0] for p in pops]
    udas = [alg.extract(GA_based_optimize) for alg in algos]
    probabs = [(uda.cm, uda.pm) for uda in udas]
    for i, best_f, probab in zip(range(len(probabs)), pops_best_f, probabs):
        print('Run {}'.format(i+1))
        print('\tMin Capital cost: {} × 1000$, cm: {}, pm: {}'.format(
            best_f, probab[0], probab[1]
        ))
    return


"""
------------------------------------------------------------------------------
"""
"""
-------------------------- Input Data -------------------------- 
"""
# Demand scenario information
# Duration of the DP (years)
T = 1.5
# Demand rate for each part (parts/hour)
C_j = pd.Series(
    data=np.array([120, 180]),
    index=['A', 'B'],
)
# Annual interest rate (%)
I = 0.12

# Part processing information
# List of OCs for every part
O_j_OC = [
    ['OC'+str(i) for i in [1, 2, 3, 4, 5, '06', 7]],
    ['OC'+str(i) for i in range(1, 12)]
]
# List of all OCs
OC_all = str_sort_num(O_j_OC)
if OC_all.index('OC06') < OC_all.index('OC6'):
    OC06_ind = OC_all.index('OC06')
    OC6_ind = OC_all.index('OC6')
    (OC_all[OC06_ind], OC_all[OC6_ind]) = (OC_all[OC6_ind], OC_all[OC06_ind])
OC_all_id = {key:value for key, value in zip(OC_all, range(len(OC_all)))}
O_j = copy.deepcopy(O_j_OC)
for i in range(len(O_j)):
    for j in range(len(O_j[i])):
        O_j[i][j] = OC_all_id[O_j[i][j]]

# OC precedence graphs
edges = [
    [
        (O_j_OC[0][0], O_j_OC[0][1]), (O_j_OC[0][0], O_j_OC[0][3]),
        (O_j_OC[0][0], O_j_OC[0][4]), (O_j_OC[0][0], O_j_OC[0][6]),
        (O_j_OC[0][1], O_j_OC[0][4]), (O_j_OC[0][4], O_j_OC[0][5]),
        (O_j_OC[0][3], O_j_OC[0][2]), (O_j_OC[0][6], O_j_OC[0][2]),
    ],
    [
        (O_j_OC[1][0], O_j_OC[1][1]), (O_j_OC[1][0], O_j_OC[1][3]),
        (O_j_OC[1][0], O_j_OC[1][4]), (O_j_OC[1][0], O_j_OC[1][6]),
        (O_j_OC[1][0], O_j_OC[1][7]), (O_j_OC[1][1], O_j_OC[1][4]),
        (O_j_OC[1][3], O_j_OC[1][2]), (O_j_OC[1][3], O_j_OC[1][8]),
        (O_j_OC[1][4], O_j_OC[1][5]), (O_j_OC[1][4], O_j_OC[1][10]),
        (O_j_OC[1][6], O_j_OC[1][2]), (O_j_OC[1][6], O_j_OC[1][9]),
        (O_j_OC[1][0], O_j_OC[1][3]),
    ]
]
G_j_OC = [nx.DiGraph(edge) for edge in edges]
G_j = [nx.relabel_nodes(i, OC_all_id) for i in G_j_OC]
# List of OSs for every part
S_j_OC = [
    ['OS'+str(i) for i in [1, 2, 3, 4, 5, '06', 7, 14, 15]],
    ['OS'+str(i) for i in range(1, 18)],
]
# List of all OS
OS_all = str_sort_num(S_j_OC)
if OS_all.index('OS06') < OS_all.index('OS6'):
    OS06_ind = OS_all.index('OS06')
    OS6_ind = OS_all.index('OS6')
    (OS_all[OS06_ind], OS_all[OS6_ind]) = (OS_all[OS6_ind], OS_all[OS06_ind])
OS_all_id = {key:value for key, value in zip(OS_all, range(len(OS_all)))}
S_j = copy.deepcopy(S_j_OC)
for i in range(len(S_j)):
    for j in range(len(S_j[i])):
        S_j[i][j] = OS_all_id[S_j[i][j]]
# Dictionary of composite OSs
OS_comp = {
    'OS12': ['OC'+str(i) for i in [3, 11]],
    'OS13': ['OC'+str(i) for i in [8, 10]],
    'OS14': ['OC'+str(i) for i in [2, 4, 7]],
    'OS15': ['OC'+str(i) for i in [2, 3, 4, 7]],
    'OS16': ['OC'+str(i) for i in [2, 4, 7, 8, 10]],
    'OS17': ['OC'+str(i) for i in [2, 3, 4, 7, 8, 10]],
}
OS_comp_id = {
    OS_all_id[key]:[
        OC_all_id[value] for value in values
    ] for key, values in OS_comp.items()
}
# Composite OSs for each part
S_j_comp = [
    [i for i in j if i in list(OS_comp_id.keys())] for j in S_j
]

# Machining information
# List of all available machine types MC
R = ['MC'+str(i) for i in range(11, 16)] + ['MC'+str(i) for i in range(21, 25)]
# Initial cost of each machine type at the beginning of DP ($ × 1000)
fi_r = pd.Series(
    data=np.array(
        [860, 1140, 1420, 1700, 1010, 385, 555, 725, 895],
        dtype=np.float
    ),
    index=R
)
# Depreciation rate (%)
D = 0.1
# Production time (hour/part)
tau_sji_r = pd.DataFrame(
    data=np.array([
        [1./120, 1./240, 1./360, 1./480, 1./120, np.inf, np.inf, np.inf,
         np.inf],
        [1./180, 1./360, 1./540, 1./720, 1./180, np.inf, np.inf, np.inf,
         np.inf],
        [1./120, 1./240, 1./360, 1./480, 1./120, 1./120, 1./240, 1./360,
         1./480],
        [1./180, 1./360, 1./540, 1./720, 1./180, np.inf, np.inf, np.inf,
         np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./60, np.inf, np.inf, np.inf, np.inf],
        [1./30, 1./60, 1./90, 1./120, 1./30, 1./30, 1./60, 1./90, 1./120],
        [1./40, 1./80, 1./120, 1./160, 1./40, 1./40, 1./80, 1./120, 1./160],
        [1./200, 1./400, 1./600, 1./800, 1./200, np.inf, np.inf, np.inf,
         np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./180, np.inf, np.inf, np.inf,
         np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./90, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./200, np.inf, np.inf, np.inf,
         np.inf],
        [1./150, 1./300, 1./450, 1./600, 1./150, np.inf, np.inf, np.inf,
         np.inf],
        [1./60, 1./120, 1./180, 1./240, 1./60, np.inf, np.inf, np.inf,
         np.inf],
        [1./120, 1./240, 1./360, 1./480, 1./120, np.inf, np.inf, np.inf,
         np.inf],
        [1./90, 1./180, 1./270, 1./360, 1./90, np.inf, np.inf, np.inf, np.inf],
        [1./60, 1./120, 1./180, 1./240, 1./60, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./60, np.inf, np.inf, np.inf,
         np.inf],
        [np.inf, np.inf, np.inf, np.inf, 1./40, np.inf, np.inf, np.inf,
         np.inf],
    ]),
    index=OS_all,
    columns=R
)
# Manufacturability of OS on machine type (0, 1)
i = np.zeros_like(tau_sji_r.values, dtype=np.int)
i[tau_sji_r.values != np.inf] = 1
Q_sji_r = pd.DataFrame(
    data=i,
    index=OS_all,
    columns=R
)

# Space limitation
# Configuration length (maksimum number of stages)
Nmax = 10
# Configuration width (maksimum number of parallel machines per stage)
Mmax = 5

# Investment limitation
# Maximum allowable initial investment of the configuration ($ × 1000)
Imax = 30000.

"""
------------------------------------------------------------------------------
"""
"""
-------------------------- Algorithm Data -------------------------- 
"""
# Population size
pop_size = 40

# Random number generation seed
rnd_seed = None

# Crossover probability
cm = 0.3

# Mutation probability
pm = 0.5

# Maximum generations
max_gen = 1000

# Logging level
log_lvl = 1

# Mutation probability list
pm_lst = np.linspace(start=0.05, stop=0.7, num=4)

# Crossover probability list
cm_lst = np.linspace(start=0.05, stop=0.7, num=5)

# Number of islands
isl_num = pm_lst.shape[0] * cm_lst.shape[0]

"""
------------------------------------------------------------------------------
"""

if __name__ == '__main__':
    # Spremanje ulaznih podatak u csv datoteke
    tau_sji_r.to_csv('production_time.csv')
    Q_sji_r.to_csv('manufacturability.csv')
    fi_r.to_csv('machine_costs.csv')

    # Test FOA for optim_config_seeking
    FOA_test = pd.DataFrame(
        data=np.array([
            [0, 15, 4, 6, -1, -1, -1, 0, 14, 2, 9, 4, 5, 11, 13, -1, -1, -1],
            [0, 0, 2, 4, 5, 5, 8, 0, 0, 0, 1, 3, 4, 5, 5, 8, 8, 8],
            [0 for i in range(7)] + [1 for i in range(11)]
            ]
        ),
        index=['OS ID', 'Stage ID', 'Part ID']
    )
    # Test chromosome
    chromosome_test = pd.DataFrame(
        data=np.array([
            [1, 2, 6, 3, 7, 4, 5, 0, 1, 0, 0, 2, 4, 5, 5, 8,
             8, 10, 6, 4, 3, 11, 9, 1, 5, 2, 7, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1,
             3, 4, 5, 5, 8, 8, 8],
        ]),
        index=['Genes']
    )

    # Problem instance and variables
    prob_var = MPFL_configuration(
        T, C_j.values, I, O_j, G_j, S_j_OC, S_j_comp, OS_comp_id, OS_all,
        R, fi_r.values, D, tau_sji_r.values, Q_sji_r.values, Nmax, Mmax,
        Imax
    )
    prob = pg.problem(prob_var)
    GenerateIndividual = generate_individual(
        prob_var.left_size, prob_var.mid_size, prob_var.right_size,
        prob_var.Nmax, seed=rnd_seed
    )
    # Population instance with initial values
    pop = pg.population(prob)
    [pop.push_back(next(GenerateIndividual)) for _ in range(pop_size)]
    # Algorithm instance
    algo = pg.algorithm(GA_based_optimize(
        pm, cm, prob_var.left_ind, prob_var.mid_ind, prob_var.right_ind,
        gen=max_gen, seed=rnd_seed, log_lvl=log_lvl
    ))
    # Generating isl_num initial populations
    pop_lst = [pg.population(prob) for _ in range(isl_num)]
    [[p.push_back(next(GenerateIndividual)) for p in pop_lst]
     for _ in range(pop_size)]
    # Generating pm_lst × cm_lst algorithms with different crossover and
    # mutation probability
    alg_lst = [pg.algorithm(GA_based_optimize(
        p, c, prob_var.left_ind, prob_var.mid_ind, prob_var.right_ind,
        gen=max_gen, seed=rnd_seed, log_lvl=log_lvl
    )) for c in cm_lst for p in pm_lst]
    # Creating islands
    islands = [pg.island(algo=a, pop=p, udi=pg.mp_island()) for a, p in
               zip(alg_lst, pop_lst)]
