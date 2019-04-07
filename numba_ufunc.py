import numba as nb

@nb.vectorize(
    [nb.int64(nb.int64, nb.int64)],
    nopython=True,
)
def vec_log_and(a, b):
    return a and b
