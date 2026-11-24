"""Data streams with known change points and anomaly labels.

Every stream returns a dict with:
  fit   - clean normal rows used to train the frozen detector (before the stream starts)
  cal   - clean normal rows used to set the starting threshold (also before the stream)
  X, y  - the stream itself, in time order (y = 1 for anomaly)
  plus a few fields that say where the changes are.
"""
import os
import numpy as np

D = 8              # number of features in the synthetic streams
T = 8000           # stream length (points)
T0 = 2000          # where the change starts
SHIFT = 3.0        # size of the shift of normal data, in standard deviations
ANOM_RATE = 0.02   # share of single-point anomalies
ANOM_OFFSET = 4.5  # how far a point anomaly sits from the current normal mean
N_FIT, N_CAL = 2000, 1000


def shift_profile(kind, t=np.arange(T), ramp=1500):
    """How far the normal mean has moved at each time, from 0 (start) to 1 (fully shifted).
    `ramp` is the length of the gradual move in points (1,500 in the first note)."""
    m = np.zeros(len(t))
    if kind == "sudden":
        m[t >= T0] = 1.0
    elif kind == "gradual":
        m = np.clip((t - T0) / float(ramp), 0.0, 1.0)  # linear ramp over `ramp` points
    elif kind == "recurring":
        for start in (2000, 4000, 6000):              # on for 1,000 points, off for 1,000
            m[(t >= start) & (t < start + 1000)] = 1.0
    elif kind in ("control", "sustained"):
        pass
    else:
        raise ValueError(kind)
    return m


def synthetic(kind, seed, ramp=1500):
    """Gaussian normal data whose mean moves along one direction.

    kind = control | sudden | gradual | recurring | sustained.
    `ramp` sets how many points the gradual move takes (only used when kind = gradual).
    'sustained' keeps normal data still and inserts a 1,000-point anomaly block at
    t = 4000..4999 that looks exactly like the sudden shift (same direction, same size),
    but is labeled anomalous. It tests whether an adaptive rule absorbs a sustained anomaly.
    """
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=D)
    direction /= np.linalg.norm(direction)
    fit = rng.normal(size=(N_FIT, D))
    cal = rng.normal(size=(N_CAL, D))

    t = np.arange(T)
    m = shift_profile(kind, t, ramp)
    mean = np.outer(m * SHIFT, direction)
    X = mean + rng.normal(size=(T, D))
    y = np.zeros(T, dtype=int)

    # single-point anomalies: current normal mean + noise + a big push in a random direction
    is_anom = rng.random(T) < ANOM_RATE
    pushes = rng.normal(size=(is_anom.sum(), D))
    pushes /= np.linalg.norm(pushes, axis=1, keepdims=True)
    X[is_anom] += ANOM_OFFSET * pushes
    y[is_anom] = 1

    block = None
    if kind == "sustained":
        block = (4000, 5000)
        sl = slice(*block)
        X[sl] = SHIFT * direction + rng.normal(size=(block[1] - block[0], D))
        y[sl] = 1
    return dict(fit=fit, cal=cal, X=X, y=y, m=m, kind=kind, block=block)


# ---------------------------------------------------------------- real data
GAS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gas", "Dataset")
ANOMALY_GAS = 4     # acetaldehyde rows are the anomalies; the other 5 gases are normal
GAS_ANOM_RATE = 0.03


def _read_batch(i):
    labels, rows = [], []
    with open(os.path.join(GAS_DIR, f"batch{i}.dat")) as f:
        for line in f:
            parts = line.split()
            labels.append(int(parts[0].split(";")[0]))
            row = np.zeros(128)
            for p in parts[1:]:
                k, v = p.split(":")
                row[int(k) - 1] = float(v)
            rows.append(row)
    return np.array(labels), np.array(rows)


def gas(seed):
    """UCI Gas Sensor Array Drift (Vergara et al., 2012): 10 batches over 36 months.

    Batch 1 normal rows train and calibrate the frozen detector. Batches 2-10 are streamed
    in time order (rows shuffled inside each batch, since files are grouped by gas).
    Acetaldehyde rows are sampled at 3% of each batch's normal rows and marked anomalous.
    The sensors age between batches, so normal data drifts with no injected change.
    """
    rng = np.random.default_rng(seed)
    lab1, X1 = _read_batch(1)
    normal1 = X1[lab1 != ANOMALY_GAS]
    idx = rng.permutation(len(normal1))
    n_fit = int(0.6 * len(normal1))
    fit_raw, cal_raw = normal1[idx[:n_fit]], normal1[idx[n_fit:]]

    def prep(A):  # signed log to tame the huge raw ranges
        return np.sign(A) * np.log1p(np.abs(A))

    mu, sd = prep(fit_raw).mean(0), prep(fit_raw).std(0) + 1e-8
    norm = lambda A: (prep(A) - mu) / sd

    anom1 = norm(X1[lab1 == ANOMALY_GAS])   # batch-1 anomalies: separation check only

    Xs, ys, batch = [], [], []
    for b in range(2, 11):
        lab, Xb = _read_batch(b)
        nor = Xb[lab != ANOMALY_GAS]
        an = Xb[lab == ANOMALY_GAS]
        n_an = min(len(an), max(1, int(round(GAS_ANOM_RATE * len(nor)))))
        an = an[rng.choice(len(an), n_an, replace=False)]
        Xb2 = np.vstack([nor, an])
        yb = np.r_[np.zeros(len(nor), int), np.ones(n_an, int)]
        p = rng.permutation(len(yb))
        Xs.append(norm(Xb2[p])); ys.append(yb[p]); batch.append(np.full(len(yb), b))
    return dict(fit=norm(fit_raw), cal=norm(cal_raw), X=np.vstack(Xs), y=np.concatenate(ys),
                batch=np.concatenate(batch), kind="gas", anom1=anom1)
