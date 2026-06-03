import numpy as np

try:
    from fastdtw import fastdtw
except Exception:  # pragma: no cover - optional dependency
    fastdtw = None

try:
    from scipy.spatial.distance import euclidean
except Exception:  # pragma: no cover - optional dependency
    euclidean = None


def _as_float_array(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _keogh_envelope(series: np.ndarray, radius: int):
    arr = _as_float_array(series)
    window = max(0, int(radius))
    upper = np.empty_like(arr)
    lower = np.empty_like(arr)

    for idx in range(len(arr)):
        start = max(0, idx - window)
        stop = min(len(arr), idx + window + 1)
        slice_ = arr[start:stop]
        upper[idx] = np.max(slice_)
        lower[idx] = np.min(slice_)

    return lower, upper


def _lb_keogh(candidate: np.ndarray, reference: np.ndarray, radius: int) -> float:
    candidate_arr = _as_float_array(candidate)
    reference_arr = _as_float_array(reference)
    if len(candidate_arr) == 0 or len(reference_arr) == 0:
        return 0.0

    lower, upper = _keogh_envelope(reference_arr, radius)
    limit = min(len(candidate_arr), len(lower))
    lb = 0.0
    for idx in range(limit):
        value = candidate_arr[idx]
        if value > upper[idx]:
            lb += (value - upper[idx]) ** 2
        elif value < lower[idx]:
            lb += (value - lower[idx]) ** 2
    return float(np.sqrt(lb))


def _banded_dtw_distance(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    a_arr = _as_float_array(a)
    b_arr = _as_float_array(b)
    n = len(a_arr)
    m = len(b_arr)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return float('inf')

    band = max(int(radius), abs(n - m))
    prev = np.full(m + 1, np.inf, dtype=float)
    curr = np.full(m + 1, np.inf, dtype=float)
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr.fill(np.inf)
        j_start = max(1, i - band)
        j_stop = min(m, i + band)
        for j in range(j_start, j_stop + 1):
            cost = abs(a_arr[i - 1] - b_arr[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return float(prev[m])


def dtw_distance(a: np.ndarray, b: np.ndarray, radius: int = 1, cutoff: float | None = None):
    """Compute DTW distance between two 1D arrays.

    A lightweight LB_Keogh screen is applied first when `cutoff` is provided.
    If the lower bound already exceeds the cutoff, the candidate is pruned and
    `inf` is returned without evaluating fastdtw.
    """
    a_arr = _as_float_array(a)
    b_arr = _as_float_array(b)
    if cutoff is not None:
        lower_bound = _lb_keogh(a_arr, b_arr, radius)
        if lower_bound > cutoff:
            return float('inf')
    if fastdtw is not None:
        try:
            if euclidean is not None:
                return fastdtw(a_arr, b_arr, dist=euclidean, radius=max(1, int(radius)))[0]
            return fastdtw(a_arr, b_arr, radius=max(1, int(radius)))[0]
        except Exception:
            pass
    return _banded_dtw_distance(a_arr, b_arr, radius=max(1, int(radius)))


def multivariate_dtw(a: np.ndarray, b: np.ndarray, radius: int = 1, cutoff: float | None = None):
    """Compute the sum of DTW distances across columns for multivariate series."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError('multivariate_dtw expects 2D arrays')
    if a_arr.shape[1] != b_arr.shape[1]:
        raise ValueError('Input arrays must have the same number of columns')

    total = 0.0
    for i in range(a_arr.shape[1]):
        column_cutoff = None if cutoff is None else max(0.0, cutoff - total)
        distance = dtw_distance(a_arr[:, i], b_arr[:, i], radius=radius, cutoff=column_cutoff)
        if np.isinf(distance):
            return float('inf')
        total += distance
    return total
