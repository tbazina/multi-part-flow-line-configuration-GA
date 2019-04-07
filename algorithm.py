import numpy as np
import numba as nb


def crossover(i_a, i_b, cm, left_ind, mid_ind):
    # Left part point-based crossover
    i_c, i_d = i_a.copy(), i_b.copy()
    mask_prob = [np.random.random(len(l)) for l in left_ind]
    mask = [np.array(l)[m<=cm] for l, m in zip(left_ind, mask_prob)]
    # print('Mask: {}'.format(mask))
    mask_inv = [np.array(l)[m>cm] for l, m in zip(left_ind, mask_prob)]
    # print('Mask inverse: {}'.format(mask))
    for m, m_i, l in zip(mask, mask_inv, left_ind):
        i_c[m_i] = i_b[l][
            np.isin(i_b[l], i_a[m], assume_unique=True, invert=True)
        ]
        i_d[m_i] = i_a[l][
            np.isin(i_a[l], i_b[m], assume_unique=True, invert=True)
        ]

    # Middle part two-point crossover for binary encoding
    if np.random.random() <= cm:
        two_pt = [
            np.sort(np.random.choice(
                m, size=2, replace=False
            )) for m in mid_ind
        ]
        # print('Two points: {}'.format(two_pt))
        for pts in two_pt:
            i_c[pts[0]:pts[1]+1] = i_b[pts[0]:pts[1]+1]
            i_d[pts[0]:pts[1]+1] = i_a[pts[0]:pts[1]+1]
    return i_c, i_d


def mutation(i_c, i_d, pm, left_ind, mid_ind, right_ind):
    # Left part swap mutation for permutation encoding
    if np.random.random() <= pm:
        swp_pts = [
            np.random.choice(l, size=2, replace=False) for l in left_ind
        ]
        # print('Swap points: {}'.format(swp_pts))
        for pts in swp_pts:
            i_c[pts[0]], i_c[pts[1]] = i_c[pts[1]], i_c[pts[0]]
    if np.random.random() <= pm:
        swp_pts = [
            np.random.choice(l, size=2, replace=False) for l in left_ind
        ]
        # print('Swap points: {}'.format(swp_pts))
        for pts in swp_pts:
            i_d[pts[0]], i_d[pts[1]] = i_d[pts[1]], i_d[pts[0]]

    # Middle part mutation
    mask_prob = [np.random.random(len(mid)) for mid in mid_ind]
    mask = [np.array(mid)[m<=pm] for mid, m in zip(mid_ind, mask_prob)]
    for m in mask:
        i_c[m] = 1 - i_c[m]
    mask_prob = [np.random.random(len(mid)) for mid in mid_ind]
    mask = [np.array(mid)[m<=pm] for mid, m in zip(mid_ind, mask_prob)]
    for m in mask:
        i_d[m] = 1 - i_d[m]

    # Right part mutation
    if np.random.random() <= pm:
        mask = [
            np.random.choice(
                r[1:-1], size=len(r)//2, replace=False) for r in right_ind
        ]
        for m in mask:
            # Odd numbers
            mask_odd = (i_c[m] % 2).astype(np.bool_)
            # print("mask :{}".format(m))
            # print("Mask odd: {}".format(mask_odd))
            # print("i_c[m]: {}".format(i_c[m]))
            # print("Odd: {}".format(i_c[m[mask_odd]]))
            if np.any(mask_odd):
                # print("Successor: {}".format(i_c[m+1][mask_odd]))
                # print("Changed values: {}".format(np.where(
                #     i_c[m[mask_odd]] < i_c[m+1][mask_odd],
                #     i_c[m[mask_odd]]+1,
                #     i_c[m[mask_odd]]
                # )))
                i_c[m[mask_odd]] = np.where(
                    i_c[m[mask_odd]] < i_c[m+1][mask_odd],
                    i_c[m[mask_odd]]+1,
                    i_c[m[mask_odd]]
                )
            # print("Odd: {}".format(i_c[m[mask_odd]]))
            # Even numbers
            mask_even = np.invert(mask_odd)
            # print("Even: {}".format(i_c[m[mask_even]]))
            if np.any(mask_even):
                # print("Predecessor: {}".format(i_c[m-1][mask_even]))
                i_c[m[mask_even]] = np.where(
                    i_c[m[mask_even]] > i_c[m-1][mask_even],
                    i_c[m[mask_even]]-1,
                    i_c[m[mask_even]]
                )
    if np.random.random() <= pm:
        mask = [
            np.random.choice(
                r[1:-1], size=len(r)//2, replace=False) for r in right_ind
        ]
        for m in mask:
            # Odd numbers
            mask_odd = (i_d[m] % 2).astype(np.bool_)
            if np.any(mask_odd):
                i_d[m[mask_odd]] = np.where(
                    i_d[m[mask_odd]] < i_d[m+1][mask_odd],
                    i_d[m[mask_odd]]+1,
                    i_d[m[mask_odd]]
                )
            # Even numbers
            mask_even = np.invert(mask_odd)
            if np.any(mask_even):
                i_d[m[mask_even]] = np.where(
                    i_d[m[mask_even]] > i_d[m-1][mask_even],
                    i_d[m[mask_even]]-1,
                    i_d[m[mask_even]]
                )
    return i_c, i_d



class GA_based_optimize:
    def __init__(self, pm, cm, left_ind, mid_ind, right_ind, gen=1,
                 seed=None, log_lvl=1):
        self.seed = seed
        self.pm = pm
        self.cm = cm
        self.left_ind = left_ind
        self.mid_ind = mid_ind
        self.right_ind = right_ind
        self.gen = gen
        self.log_lvl = log_lvl
        self.log = []

        # Random number seed
        np.random.seed(self.seed)

    def evolve(self, pop):
        for gen in range(self.gen):
            # Old population
            old_pop_x = pop.get_x()
            old_pop_f = pop.get_f()
            # New population
            new_pop_x = np.empty_like(old_pop_x, dtype=np.float64)

            for i in range(len(pop)//2):
                # Roulette wheel selection
                fit = 1. / old_pop_f[:, 0]
                fit_sum = fit.sum()
                prob = fit / fit_sum
                ind = np.random.choice(prob.size, size=2, p=prob)
                i_a, i_b = old_pop_x[ind]

                # Crossover left and middle part
                i_c, i_d = crossover(
                    i_a, i_b, self.cm, self.left_ind, self.mid_ind
                )

                # Mutation
                new_pop_x[i*2:i*2+2] = mutation(
                    i_c, i_d, self.pm, self.left_ind, self.mid_ind,
                    self.right_ind
                )
            # Fitness of new population
            new_pop_f = np.array(
                [pop.problem.fitness(i) for i in new_pop_x],
                dtype=np.float64
            )
            # mi + lambda selection for maintaining high quality of solutions
            both_pop_x = np.vstack((old_pop_x, new_pop_x))
            both_pop_f = np.vstack((old_pop_f, new_pop_f))
            # Sortirani indices
            sorted_idx = np.lexsort(np.hstack((both_pop_x, both_pop_f)).T)
            # Unique vrijednosti
            row_mask = np.append(
                [True], np.any(np.diff(both_pop_x[sorted_idx, :], axis=0),
                               axis=1)
            )
            # Populacija nakon selekcije
            [pop.set_xf(i, x, f) for i, x, f in zip(
                range(len(pop)), both_pop_x[sorted_idx[row_mask]],
                both_pop_f[sorted_idx[row_mask]])]
            # Log Generation, Min cost, Average cost
            if not (gen % self.log_lvl):
                self.log.append((
                    gen,
                    both_pop_f[sorted_idx[row_mask]][0],
                    both_pop_f[sorted_idx[row_mask]][:len(pop)].mean()
                ))
        return pop

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)

    def set_verbosity(self, level):
        self.log_lvl = level

    def get_log(self):
        return self.log

    def get_name(self):
        return 'GA-based optimisation'

    def get_extra_info(self):
        return '\n\t'.join(
            [
                '\tGenerations: {}'.format(self.gen),
                'Mutation probability pm: {}'.format(self.pm),
                'Crossover probability cm: {}'.format(self.cm),
                'Seed: {}'.format(self.seed),
                'Log level: {}'.format(self.log_lvl),
                'Selection for Crossover: {}'.format('Roulette wheel'),
                'Left part Crossover: {}'.format('Position-based crossover for'
                                                 ' permutation encoding'),
                'Middle part Crossover: {}'.format('Two-point crossover for '
                                                   'binary encoding'),
                'Right part Crossover: {}'.format(None),
                'Left part Mutation: {}'.format('Swap mutation for permutation'
                                                ' encoding'),
                'Middle part Mutation: {}'.format('Random select and reverse'
                                                  ' value'),
                'Right part Mutation: {}'.format('Select half and change '
                                                 'according to condition'),
                'Generation Selection: {}'.format('mu + lambda selection to '
                                                  'maintain the high quality '
                                                  'of solutions'),
            ]
        )