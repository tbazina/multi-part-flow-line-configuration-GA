import numpy as np


def generate_individual(left_size, mid_size, right_size, Nmax, seed=None):
    """
    Generator for random generation of individual decision vectors. Constraints
    for letf, middle and right parts are satisfied.

    :param left_size:
    List of left chromosome part sizes for each part in production.

    :param mid_size:
    List of middle chromosome part sizes for each part in production.

    :param right_size:
    List of right chromosome part sizes for each part in production.

    :param Nmax:
    Maximum number of stages

    :param seed:
    Seed for random number generator.

    :yield:
    np.array Randomly generated decision vector
    """
    np.random.seed(seed=seed)
    while True:
        left = [np.random.permutation(l)+1 for l in left_size]
        mid = [
            np.random.randint(low=2, size=m, dtype=np.int64) for m in mid_size
        ]
        right = [
            np.sort(np.random.randint(low=Nmax, size=r-2)) for r in right_size
        ]
        right = [np.insert(r, 0, 0) for r in right]
        right = [np.append(r, Nmax-1) for r in right]
        res = [[left[i], mid[i], right[i]] for i in range(len(left))]
        yield np.hstack([item for sublist in res for item in sublist])
