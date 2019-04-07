import pygmo as pg
import numpy as np
import numba as nb
import networkx as nx


@nb.jit(
    [
        nb.int64[:](nb.int64[:]),
        nb.float64[:](nb.float64[:])
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def np_unique_impl(a):
    b = np.sort(a.ravel())
    head = list(b[:1])
    tail = [x for i, x in enumerate(b[1:]) if b[i] != x]
    return np.array(head + tail)


@nb.jit(
    [
        nb.float64(
            nb.int64, nb.int64[:, :], nb.float64[:, :], nb.int64[:], nb.int64,
            nb.float64[:], nb.float64, nb.int64, nb.int64[:, :]
    )],
    nopython=True,
    cache=True,
    fastmath=True,
)
def optimal_config_seeking_nb(
        Nmax, Q_sji_r, tau_sji_r, C_j, Mmax, fi_r, CI, Imax, FOA
):
    """
    Obtaining optimal configuration associated with FOA. Numba implementation.

    :param Nmax:
    Maximum number of stages

    :param Q_sji_r:
    Manufacturability of OS s_ji with machine type r

    :param tau_sji_r:
    Time [h] for performing OS s_ji on machine r

    :param C_j:
    Necessary production capacity [part/h] for each part

    :param Mmax:
    Maximum number of parallel machines per stage

    :param fi_r:
    Initial cost [$×1000] of machine type r at the beginning of DP

    :param CI:
    Capital investment factor (Depreciation rate and Annual interest rate)

    :param Imax:
    Maximum allowable initial investment of the configuration

    :param FOA:
    2 dim Feasible OS assignment np.array (ID OS, ID stage, ID part), -1 for
    non-existing OS.

    :return: Copt
    Capital cost of the optimal configuration for given FOA
    """

    # Indices OS-a za svaki stage
    os_shape = 0
    num_pr = C_j.shape[0]
    for k in range(Nmax):
        for p in range(num_pr):
            mask = np.logical_and(
                np.logical_and(FOA[1, :] == k, FOA[0, :] >= 0),
                FOA[2, :] == p
            )
            if mask[mask == np.bool_(1)].shape[0] > os_shape:
                os_shape = mask[mask == np.bool_(1)].shape[0]
    # Array stage, part, OS ID
    stages = np.full((Nmax, num_pr, os_shape), -1, np.int64)
    # Usable machine types for each stage (can perform all OS-s)
    machine = np.full((Nmax, fi_r.shape[0]), -1, np.int64)
    # Usable machine numbers
    machine_num = np.full_like(machine, -1, np.int64)
    # Usable machine cost
    machine_cost = np.full_like(machine, -1, np.float64)
    for k in range(Nmax):
        for p in range(num_pr):
            # Samo stage-ovi gdje postoje OS-ovi
            mask = np.logical_and(
                np.logical_and(FOA[1, :] == k, FOA[0, :] >= 0),
                FOA[2, :] == p
            )
            if np.any(mask):
                os_ids = FOA[0, :][mask]
                for i in range(os_ids.shape[0]):
                    stages[k, p, i] = os_ids[i]
    # print('stages')
    # print(stages)
    for k in range(Nmax):
        # Array OS ID za pojedini stage s ponavljanjem i jedinstveni
        stg_os_flat = np.ravel(stages[k, :, :])
        stg_os = stg_os_flat[stg_os_flat >= 0]
        if np.any(stg_os>=0):
            stg_os_unq = np_unique_impl(stg_os)
            Q_sji_r_ops = Q_sji_r[stg_os_unq]
            # Array 0 (ne mogu) i 1 (mogu) popisa strojeva
            machine_bool = np.ones(Q_sji_r_ops.shape[1], np.int64)
            for i in range(Q_sji_r_ops.shape[0]):
                machine_bool = np.logical_and(
                    machine_bool,
                    Q_sji_r_ops[i],
                ).astype(np.int64)
            # Indices of machine types in stage
            mach_type = np.nonzero(machine_bool)[0]
            if np.any(mach_type):
                # OS times
                tau = tau_sji_r[stg_os][:, mach_type]
                # Necessary production capacity
                c_i = np.empty(tau.shape[0], np.int64)
                num_fill = 0
                for p in range(num_pr):
                    os_size = stages[k, p, :][stages[k, p, :] >= 0].shape[0]
                    p_fill = np.full(os_size, C_j[p], np.int64)
                    c_i[num_fill:num_fill+os_size] = p_fill
                    num_fill += os_size
                # Minimal number of machines in each stage for each usable
                # machine type
                n_rk = np.ceil(
                    np.transpose(tau.transpose() * c_i).sum(0)
                ).astype(np.int64)
                # Penalty value for more machines than Mmax
                PW = 1. + (n_rk - Mmax) / Mmax
                PW[PW < 1.] = 1.
                # Cost of usable machines in every workstation
                cost = n_rk * fi_r[mach_type] * CI * PW
                for i in range(mach_type.shape[0]):
                    # Indices stroja
                    machine[k, i] = mach_type[i]
                    # Minimalno potrebno strojeva
                    machine_num[k, i] = n_rk[i]
                    # Trošak strojeva za svaki stage
                    machine_cost[k, i] = cost[i]
    # print('machine')
    # print(machine)
    # print('machine_num')
    # print(machine_num)
    # print('machine_cost')
    # print(machine_cost)

    # Capital cost of optimal MPFL configuration
    C_FL = 0.
    for k in range(Nmax):
        # Array troskova po stage-u
        cost = machine_cost[k, :][machine_cost[k, :] >= 0.]
        if np.any(cost):
            min_ind = cost.argmin()
            C_FL += cost[min_ind]
            # Ispis rezultata
            # print(
            #     ('Stage: {}; Machine type: {}; Number of machines: {}; '
            #      'Stage cost: {}').format(
            #         k, machine[k, :][min_ind], machine_num[k, :][min_ind],
            #         machine_cost[k, :][min_ind]
            #     )
            # )
    # Penalty item for investment larger than Imax
    PI = 1. + (C_FL/CI - Imax) / Imax
    if PI < 1.: PI = 1.
    Copt = C_FL * PI
    return Copt


@nb.jit(
    [
        nb.int64[:](nb.int64[:], nb.int64[:])
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def search_sequence_numpy(arr, seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------
    Output : 1D Array of indices in the input array that satisfy the
    matching of input sequence in the input array.
    In case of no match, empty array is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create 2D array of sliding indices across entire length of input array.
    r_arr = np.arange(Na - Nseq + 1)
    _r_arr = np.lib.stride_tricks.as_strided(
        r_arr, (r_arr.shape[0], 1), (r_arr.strides[0], 0)
    )
    sld_ind = _r_arr + r_seq

    # Match up with the input sequence & get the matching starting indices.
    M = np.empty_like(sld_ind, np.int64)
    M_all = np.empty(M.shape[0], np.int64)
    for i in range(M.shape[0]):
        M[i] = (arr[sld_ind[i]] == seq)
        M_all[i] = np.int64(np.all(M[i]))

    # Get the range of those indices as final output
    return np.where(np.convolve(M_all, np.ones((Nseq), np.int64)) > 0)[0]


def FOC_decoding(left_part, G, O):
    """
    Dekodiranje lijevih dijelova kromosoma u valid FOC.

    :param left_part:
    np.array: selection priority of OC for part

    :param G:
    nx.DiGraph: OC precedence graph for part

    :param O:
    np.array: List of OCs for part

    :return:
    np.array: viable FOC
    """

    def lexicographical_keys(node, left_part=left_part, O=O):
        """
        Mapiranje tocaka DAG-a za priority topological sort.

        :param node:
        int: ID of OC in DAG

        :param left_part:
        np.array: selection priority of OC for part

        :param O:
        np.array: List of OCs for part

        :return:
        int: selection priority of node (lower is better)
        """
        return O.size - left_part[np.where(O == node)[0][0]]

    return np.array(
        [i for i in nx.lexicographical_topological_sort(
            G, lexicographical_keys
        )]
    )



@nb.jit(
    [
        nb.int64[:, :](
            nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:], nb.int64[:, :],
            nb.int64
        )
    ],
    nopython=True,
    cache=True,
    fastmath=True,
)
def FOA_decoding(FOC, middle_part, right_part, S_comp, S_comp_OC, part_ID):
    """
    Decode Chromosome to FOS from FOC and composite OS. Decode FOC and right
    Chromosome part to FOA.

    :param FOC:
    np.array Decoded FOC

    :param middle_part:
    np.array Middle Chromosome part (composite OS)

    :param right_part:
    np.array Right Chromosome part (stages)

    :param S_comp:
    np.array Composite OS ID for part

    :param S_comp_OC:
    2D np.array Containing OC for each composite OS

    :param part_ID:
    int64 Part ID (0, 1, ...)

    :return:
    2D np.array FOA for part (OS ID, Stage ID, Part ID)
    """
    FOS = FOC.copy()
    if np.any(middle_part):
        S_subst = S_comp[middle_part.astype(np.bool_)]
        S_subst_OC = S_comp_OC[middle_part.astype(np.bool_)]
        FOS_ind_sort = np.argsort(FOS)
        for i in range(S_subst_OC.shape[0]):
            subst_ind_search = np.searchsorted(
                FOS[FOS_ind_sort], S_subst_OC[i][S_subst_OC[i] >= 0])
            if subst_ind_search.size == 0:
                continue
            subst_ind_search[subst_ind_search > FOS.size-1] = FOS.size - 1
            subst_ind = FOS_ind_sort[subst_ind_search]
            # Continue loop if some OCs are not found
            if np.any(FOS[subst_ind] != S_subst_OC[i][S_subst_OC[i] >= 0]):
                continue
            # Check if all found OCs are next to each other
            subst_ind_sort = np.sort(subst_ind)
            subst_ind_diff = np.diff(subst_ind_sort)
            if np.all(subst_ind_diff == 1):
                FOS = np.hstack(
                    (FOS[:subst_ind_sort[0]], FOS[subst_ind_sort[-1]:])
                )
                FOS[subst_ind_sort[0]] = S_subst[i]
                FOS_ind_sort = np.argsort(FOS)
    # Extending FOS to concatenate with right part and part ID
    FOS_extended = np.full_like(right_part, -1, np.int64)
    FOS_extended[:FOS.size] = FOS
    part_ID_extended = np.full_like(right_part, part_ID, np.int64)
    FOA = np.vstack((FOS_extended, right_part, part_ID_extended))
    return FOA


# Problem class
class MPFL_configuration:
    def __init__(self, T, C_j, I, O_j, G_j, S_j, S_j_comp, OS_comp_ID, OS_all,
                 R, fi_r, D, tau_sji_r, Q_sji_r, Nmax, Mmax, Imax):
        """

        :param T:
        Duration of the DP [years]

        :param C_j:
        Necessary production capacity [part/h] for each part

        :param I:
        Annual interest rate (%)

        :param O_j:
        List of OC ID for every part

        :param G_j:
        List of OC ID precedence graphs for every part

        :param S_j:
        List of OS for every part

        :param S_j_comp:
        List of composite OS ID for every part

        :param OS_comp_ID:
        Dictionary of OC ID for every composite OS ID

        :param OS_all:
        List of all OS

        :param R:
        List of all available machine types MC

        :param fi_r:
        Initial cost [$×1000] of machine type r at the beginning of DP

        :param D:
        Depreciation rate (%)

        :param tau_sji_r:
        Time [h] for performing OS s_ji on machine r

        :param Q_sji_r:
        Manufacturability of OS s_ji with machine type r

        :param Nmax:
        Maximum number of stages

        :param Mmax:
        Maximum number of parallel machines per stage

        :param Imax:
        Maximum allowable initial investment of the configuration
        """

        self.T = T
        self.C_j = C_j
        self.I = I
        self.O_j = O_j
        self.G_j = G_j
        self.S_j = S_j
        self.S_j_comp = S_j_comp
        self.OS_comp_ID = OS_comp_ID
        self.OS_all = OS_all
        self.R = R
        self.fi_r = fi_r
        self.D = D
        self.tau_sji_r = tau_sji_r
        self.Q_sji_r = Q_sji_r
        self.Nmax = Nmax
        self.Mmax = Mmax
        self.Imax = Imax

        # Capital investment factor (Depreciation rate, Annual interest rate)
        self.CI = 1. - ((1. - D)**T / (1. + I)**T)

        # List of 2D np.arrays containing OC for every composite OS for every
        # part
        self.S_j_comp_OC_list = [
            [self.OS_comp_ID[comp] for comp in pt_comp]
            for pt_comp in self.S_j_comp
        ]
        print(self.S_j_comp_OC_list)
        self.S_j_comp_OC_len = [
            len(max(self.S_j_comp_OC_list[i], key=len)) for i in range(
                len(self.S_j_comp_OC_list))
        ]
        print(self.S_j_comp_OC_len)
        self.S_j_comp_OC = [
            np.full(
                (len(self.S_j_comp_OC_list[i]), self.S_j_comp_OC_len[i]), -1,
                np.int64) for i in range(len(self.S_j_comp_OC_list))
        ]
        for i in range(len(self.S_j_comp_OC)):
            for j in range(self.S_j_comp_OC[i].shape[0]):
                self.S_j_comp_OC[i][j, :len(self.S_j_comp_OC_list[i][j])] = (
                self.S_j_comp_OC_list[i][j]
                )
        print(self.S_j_comp_OC)

        # Size of chromosome parts (decision vector)
        self.left_size = [len(i) for i in self.O_j]
        print('Left part size: {}'.format(self.left_size))
        self.right_size = [len(i) for i in self.O_j]
        print('Right part size: {}'.format(self.right_size))
        self.mid_size = [len(i) for i in self.S_j_comp]
        print('Middle part size: {}'.format(self.mid_size))

        # Indices of every chromosome side (left, right, mid) for every part
        self.left_ind = [[i for i in range(ls)] for ls in self.left_size]
        self.mid_ind = [[i for i in range(ls)] for ls in self.mid_size]
        self.right_ind = [
            [i for i in range(ls)] for ls in self.right_size
        ]
        ind_incr = 0
        for i in range(1, len(self.left_ind)):
            ind_incr += self.left_size[i-1] + self.mid_size[i-1] + \
                        self.right_size[i-1]
            self.left_ind[i] = [j + ind_incr for j in self.left_ind[i]]
        print('Left part indices: {}'.format(self.left_ind))

        for i in range(len(self.mid_ind)):
            self.mid_ind[i] = [
                j + self.left_ind[i][-1] + 1 for j in self.mid_ind[i]
            ]
        print('Middle part indices: {}'.format(self.mid_ind))

        for i in range(len(self.right_ind)):
            self.right_ind[i] = [
                j + self.mid_ind[i][-1] + 1 for j in self.right_ind[i]
            ]
        print('Right part indices: {}'.format(self.right_ind))

        # Number of parts
        self.part_num = len(self.left_size)
        self.prob_dim = \
            sum(self.left_size) + sum(self.mid_size) + sum(self.right_size)

    def fitness(self, x):
        """
        FOA_decoding algorithm is used to decode the chromosome to an FOA.
        optimal_config_seeking algorithm is used to obtain optimal
        configuration for the decoded FOA. Fitness is the capital cost of the
        configuration.

        :param x: np.array of encoded FOA

        :return:
        """
        x = x.astype(np.int64)
        left_part = [x[l_ind] for l_ind in self.left_ind]
        # print('Left part: {}'.format(left_part))
        mid_part = [x[m_ind] for m_ind in self.mid_ind]
        # print('Middle part: {}'.format(mid_part))
        right_part = [x[r_ind] for r_ind in self.right_ind]
        # print('Right part: {}'.format(right_part))

        # Obtain FOC
        FOC = [
            FOC_decoding(
                left_part[i], self.G_j[i], np.array(self.O_j[i], np.int64)
            ) for i in range(len(left_part))
        ]

        # Obtain FOA
        FOA = [
            FOA_decoding(
                FOC[i], mid_part[i], right_part[i],
                np.array(self.S_j_comp[i], np.int64), self.S_j_comp_OC[i], i
            ) for i in range(len(mid_part))
        ]

        # Calculate capital cost (fitness)
        return [optimal_config_seeking_nb(
            self.Nmax, self.Q_sji_r, self.tau_sji_r, self.C_j, self.Mmax,
            self.fi_r, self.CI, self.Imax, np.hstack(FOA)
        )]

    def get_top_ten_config(self, pops):
        # Extracting population and fitness from islands
        # pops = [isl.get_population() for isl in islands]
        pop_x = [p.get_x() for p in pops]
        pop_f = [p.get_f() for p in pops]

        # Top 10 jedinstvenih konfiguracija sa svih otoka
        # mi + lambda selection for maintaining high quality of solutions
        isl_pop_x = np.vstack(pop_x)
        isl_pop_f = np.vstack(pop_f)
        # Sortirani indices
        sorted_idx = np.lexsort(np.hstack((isl_pop_x, isl_pop_f)).T)
        # Unique vrijednosti
        row_mask = np.append(
            [True], np.any(np.diff(isl_pop_x[sorted_idx, :], axis=0),
                           axis=1)
        )
        for i in range(isl_pop_x[sorted_idx[row_mask]][:10, :].shape[0]):
            x = isl_pop_x[sorted_idx[row_mask]][:10, :][i]
            x = x.astype(np.int64)
            left_part = [x[l_ind] for l_ind in self.left_ind]
            # print('Left part: {}'.format(left_part))
            mid_part = [x[m_ind] for m_ind in self.mid_ind]
            # print('Middle part: {}'.format(mid_part))
            right_part = [x[r_ind] for r_ind in self.right_ind]
            # print('Right part: {}'.format(right_part))

            # Obtain FOC
            FOC = [
                FOC_decoding(
                    left_part[i], self.G_j[i], np.array(self.O_j[i], np.int64)
                ) for i in range(len(left_part))
            ]

            # Obtain FOA
            FOA = [
                FOA_decoding(
                    FOC[i], mid_part[i], right_part[i],
                    np.array(self.S_j_comp[i], np.int64), self.S_j_comp_OC[i], i
                ) for i in range(len(mid_part))
            ]

            print('_____________________________________________________')
            print('Configuration: {}'.format(i+1))
            # Calculate and print configurations
            self.optimal_config_seeking(np.hstack(FOA))
            print('_____________________________________________________')
        return


    def optimal_config_seeking(self, FOA):
        """
        Obtaining optimal configuration associated with FOA. Pure Python
        implementation.

        :param FOA:
        3 dim Feasible OS assignment np.array (ID OS, ID stage, ID part), -1 for
        non-existing OS.

        :return: Copt
        Capital cost of the optimal configuration for given FOA
        """
        # Indices OS-a za svaki stage
        stages = []
        # Usable machine types for each stage (can perform all OS-s)
        machines = []
        for k in range(self.Nmax):
            # Samo stage-ovi gdje postoje OS-ovi
            mask = np.logical_and(FOA[1, :] == k, FOA[0, :] >= 0)
            if np.any(mask):
                stages.append((k, FOA[0, :][mask], FOA[2, :][mask]))
                # Array 0 (ne mogu) i 1 (mogu) popisa strojeva
                machine_bool = np.logical_and.reduce(
                    self.Q_sji_r[np.unique(stages[-1][1])],
                    dtype=np.int
                )
                # Indices of machine types in stage
                mach_type = np.nonzero(machine_bool)[0]
                # OS times
                tau = self.tau_sji_r[stages[-1][1]][:, mach_type]
                # Necessary production capacity
                c_i = self.C_j[stages[-1][2]]
                # Minimal number of machines in each stage for each usable
                # machine type
                n_rk = np.ceil(
                    np.transpose(tau.transpose() * c_i).sum(0)
                ).astype(np.int)
                # Penalty value for more machines than Mmax
                PW = 1. + (n_rk - self.Mmax) / self.Mmax
                PW[PW < 1.] = 1.
                # Cost of usable machines in every workstation
                cost = n_rk * self.fi_r[mach_type] * self.CI * PW
                # Indices stroja, minimalno potrebno strojeva i trošak strojeva
                # za svaki stage
                machines.append((k, mach_type, n_rk, cost))
        # print('stages')
        # print(stages)
        # print('machines')
        # print(machines)

        # Capital cost of optimal MPFL configuration
        C_FL = 0.
        for k in range(len(machines)):
            min_ind = machines[k][3].argmin()
            C_FL += machines[k][3][min_ind]
            print(
                ('Stage: {}; Mach. type: {}; No. mach: {}; '
                 'Stage cost: {}').format(
                    machines[k][0], machines[k][1][min_ind],
                    machines[k][2][min_ind], machines[k][3][min_ind]
                )
            )
            print(
                '\tPart IDs: {}, OS IDs: {}'.format(stages[k][2], stages[k][1])
            )
        # Penalty item for investment larger than Imax
        PI = 1. + (C_FL/self.CI - self.Imax) / self.Imax
        if PI < 1.: PI = 1.
        Copt = C_FL * PI
        print('\t\t\tCapital cost: {}'.format(Copt))
        return Copt

    def call_jit_config(self, FOA):
        return optimal_config_seeking_nb(
            self.Nmax, self.Q_sji_r, self.tau_sji_r, self.C_j, self.Mmax,
            self.fi_r, self.CI, self.Imax, FOA
        )

    def get_bounds(self):
        # Lower bounds
        lb = []
        # Upper bounds
        ub = []
        for i in range(self.part_num):
            lb.extend(
                [1]*self.left_size[i] +
                [0]*self.mid_size[i] +
                [0]*self.right_size[i]
            )
            ub.extend(
                [self.left_size[i]]*self.left_size[i] +
                [1]*self.mid_size[i] +
                [self.Nmax-1]*self.right_size[i]
            )
        return (lb, ub)

    def get_nix(self):
        # Integer dimension
        return self.prob_dim

    def get_name(self):
        return 'MPFL Configuration Problem'

